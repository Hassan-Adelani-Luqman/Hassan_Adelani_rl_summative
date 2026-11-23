"""Shared training plumbing: TorchRL env factory, run configuration, evaluation, logging, checkpoints.

Every algorithm trains on the same TorchRL environment: a gymnasium vector of
``AfyaSentinelEnv`` copies wrapped by ``GymWrapper`` and transformed with
``StepCounter`` (decisions per episode), ``RewardSum`` (episode return in the env's own
units, for logging) and ``RewardScaling`` (the scaled reward the agents learn from).
Training episodes draw their seeds from ``TRAIN_SEED_POOL``; evaluation always runs on
the fixed held-out seeds with a deterministic (greedy) policy.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import tensordict
import torch
import torchrl
from tensordict import TensorDict
from torch import nn
from torchrl.data import Binary
from torchrl.data import Categorical as CategoricalSpec
from torchrl.envs import (
    ActionMask,
    Compose,
    GymWrapper,
    InitTracker,
    RewardScaling,
    RewardSum,
    StepCounter,
    Transform,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type

from analysis.baselines import DoNothing, add_infections_averted, evaluate, run_episode, summarize
from environment.custom_env import (
    A_PREVENT,
    A_SCREEN,
    ENV_VERSION,
    N_ACTIONS,
    N_VILLAGES,
    OBS_KITS,
    OBS_LOCATION,
    OBS_NETS,
    AfyaSentinelEnv,
)
from training.seeds import EVAL_SETS, TRAIN_SEED_POOL, seed_hash

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("dqn", "reinforce", "a2c", "ppo")
PG_ALGOS = ("reinforce", "a2c", "ppo")


# ------------------------------------------------------------------------------ config


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce one training run. Unused fields are ignored per algorithm."""

    algo: str
    run_id: str = "baseline"
    seed: int = 0
    total_frames: int = 300_000
    num_envs: int = 8
    scenario: str = "default"
    reward_scale: float = 0.02
    hidden: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    lr: float = 3e-4
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    # DQN
    buffer_size: int = 100_000
    batch_size: int = 128
    steps_per_collect: int = 4  # env steps per sub-env between gradient phases
    updates_per_collect: int = 4
    eps_init: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # share of the frame budget over which ε anneals
    target_update: str = "hard"  # "hard" | "soft"
    target_interval: int = 500  # optimiser steps between hard target copies
    tau: float = 0.005  # soft target update rate
    double_dqn: bool = False
    init_random_frames: int = 5_000
    # policy gradient
    rollout_len: int = 64  # steps per sub-env per update (A2C, PPO)
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    normalize_advantage: bool = True
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    episodes_per_update: int = 16  # REINFORCE
    value_baseline: bool = True  # REINFORCE: subtract a learned V(s)
    normalize_returns: bool = True  # REINFORCE
    # evaluation and outputs
    eval_sets: tuple[str, ...] = ("held_out",)
    eval_limit: int | None = None  # evaluate on the first N seeds of each set (tests only)
    logs_root: str = str(ROOT / "logs")
    models_root: str = str(ROOT / "models")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items() if k in known}
        return cls(**clean)


# Per-algorithm defaults (the "baseline" configuration of each sweep).
ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "dqn": {"lr": 5e-4, "activation": "relu", "max_grad_norm": 10.0},
    "reinforce": {"lr": 1e-3, "entropy_coef": 0.0},
    # A2C at lr 7e-4 collapsed its entropy within ~80 updates in smoke runs; 2.5e-4 learns steadily.
    "a2c": {"lr": 2.5e-4, "rollout_len": 16, "max_grad_norm": 0.5},
    "ppo": {"lr": 3e-4, "rollout_len": 128, "max_grad_norm": 0.5},
}

# Hyperparameters that matter for each algorithm (used for the report's tables).
ALGO_FIELDS: dict[str, tuple[str, ...]] = {
    "dqn": ("lr", "gamma", "buffer_size", "batch_size", "eps_end", "eps_fraction", "target_update",
            "target_interval", "tau", "double_dqn"),
    "reinforce": ("lr", "gamma", "episodes_per_update", "value_baseline", "entropy_coef", "normalize_returns"),
    "a2c": ("lr", "gamma", "rollout_len", "gae_lambda", "entropy_coef", "critic_coef", "max_grad_norm",
            "normalize_advantage"),
    "ppo": ("lr", "gamma", "clip_epsilon", "gae_lambda", "ppo_epochs", "minibatch_size", "entropy_coef",
            "rollout_len"),
}


def default_config(algo: str, **overrides: Any) -> RunConfig:
    """The baseline configuration for ``algo``, with ``overrides`` applied."""
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")
    return replace(RunConfig(algo=algo), **{**ALGO_DEFAULTS[algo], **overrides})


def run_dirs(cfg: RunConfig) -> tuple[Path, Path]:
    """(log dir, model dir): logs/<algo>/<run_id>, models/dqn/<run_id> or models/pg/<algo>/<run_id>."""
    log_dir = Path(cfg.logs_root) / cfg.algo / cfg.run_id
    family = Path("dqn") if cfg.algo == "dqn" else Path("pg") / cfg.algo
    return log_dir, Path(cfg.models_root) / family / cfg.run_id


# ------------------------------------------------------------------------------ environment


def action_mask_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """Valid actions given an observation (``[..., 55]`` -> bool ``[..., 12]``).

    Everything needed is in the observation: the van cannot drive to where it already is,
    and cannot screen or hand out nets at the hub or with none left. "Return to hub" (which
    restocks) and "Wait" are always valid. The environment still penalises invalid actions
    for any policy that ignores this mask (e.g. the random baseline).
    """
    location = obs[..., OBS_LOCATION]
    at_hub = location[..., 0] > 0.5
    mask = torch.ones(*obs.shape[:-1], N_ACTIONS, dtype=torch.bool, device=obs.device)
    mask[..., :N_VILLAGES] = location[..., 1:] < 0.5
    mask[..., A_SCREEN] = ~at_hub & (obs[..., OBS_KITS] > 0)
    mask[..., A_PREVENT] = ~at_hub & (obs[..., OBS_NETS] > 0)
    return mask


class ActionMaskFromObs(Transform):
    """Writes ``action_mask`` next to every observation (on reset and on step)."""

    def __init__(self) -> None:
        super().__init__(in_keys=["observation"], out_keys=["action_mask"])

    def _call(self, next_tensordict):
        next_tensordict.set("action_mask", action_mask_from_obs(next_tensordict.get("observation")))
        return next_tensordict

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        shape = (*observation_spec["observation"].shape[:-1], N_ACTIONS)
        observation_spec["action_mask"] = Binary(n=N_ACTIONS, shape=shape, dtype=torch.bool,
                                                 device=observation_spec.device)
        return observation_spec


class SeedPoolReset(gym.Wrapper):
    """Each episode's district is drawn from a fixed seed pool.

    Training never sees the held-out evaluation seeds, and a run is reproducible from its
    own seed: ``reset(seed=s)`` re-seeds the draw, plain ``reset()`` takes the next seed.
    """

    def __init__(self, env: gym.Env, pool: Sequence[int] = TRAIN_SEED_POOL) -> None:
        super().__init__(env)
        self._pool = pool
        self._rng = np.random.default_rng(0)
        self.episode_seed: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_seed = int(self._pool[int(self._rng.integers(len(self._pool)))])
        return self.env.reset(seed=self.episode_seed, options=options)


def make_env(
    seed: int = 0,
    scenario: str = "default",
    num_envs: int = 8,
    reward_scale: float = 0.02,
    seed_pool: Sequence[int] = TRAIN_SEED_POOL,
) -> TransformedEnv:
    """The TorchRL training environment shared by all four algorithms."""
    vector_env = gym.vector.SyncVectorEnv(
        [lambda: SeedPoolReset(AfyaSentinelEnv(scenario), seed_pool) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,  # TorchRL resets finished sub-envs itself
    )
    env = TransformedEnv(
        GymWrapper(vector_env, categorical_action_encoding=True),
        Compose(
            InitTracker(),  # marks first steps; TorchRL policies expect it from v0.16
            ActionMaskFromObs(),  # which actions are valid right now
            ActionMask(),  # random actions (e.g. DQN warm-up) respect the mask too
            StepCounter(),
            RewardSum(),  # episode_reward: the return in env units (cases), for logging
            RewardScaling(loc=0.0, scale=reward_scale),  # what the agent optimises
        ),
    )
    env.set_seed(seed)
    return env


def action_spec() -> CategoricalSpec:
    return CategoricalSpec(N_ACTIONS)


def mlp(in_features: int, out_features: int, cfg: RunConfig) -> nn.Module:
    from torchrl.modules import MLP

    activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[cfg.activation]
    return MLP(in_features=in_features, out_features=out_features, num_cells=list(cfg.hidden),
               activation_class=activation)


def episode_stats(batch: TensorDict) -> tuple[list[float], list[int]]:
    """Returns (in env units) and lengths (decisions) of the episodes that ended in a batch."""
    done = batch["next", "done"].squeeze(-1)
    returns = batch["next", "episode_reward"].squeeze(-1)[done]
    lengths = batch["next", "step_count"].squeeze(-1)[done]
    return returns.tolist(), lengths.tolist()


# ------------------------------------------------------------------------------ logging


class RunLogger:
    """Writes train.csv (one row per finished episode) and updates.csv (one row per update)."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self._writers: dict[str, tuple[Any, csv.DictWriter]] = {}
        self.episodes = 0

    def _write(self, name: str, row: dict[str, Any]) -> None:
        if name not in self._writers:
            handle = open(self.log_dir / name, "w", newline="")  # noqa: SIM115 (closed in close())
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            self._writers[name] = (handle, writer)
        self._writers[name][1].writerow(row)

    def episodes_from_batch(self, batch: TensorDict, frames: int) -> list[float]:
        returns, lengths = episode_stats(batch)
        for ret, length in zip(returns, lengths, strict=True):
            self.episodes += 1
            self._write("train.csv", {"frames": frames, "episode": self.episodes, "return": ret, "decisions": length})
        return returns

    def update(self, **row: Any) -> None:
        self._write("updates.csv", {k: (float(v) if torch.is_tensor(v) else v) for k, v in row.items()})

    def close(self) -> None:
        for handle, _ in self._writers.values():
            handle.close()
        self._writers.clear()


# ------------------------------------------------------------------------------ evaluation


def policy_mode(algo: str) -> str:
    """How each algorithm's learned policy is evaluated.

    DQN learns a greedy policy over Q-values, so it acts by argmax. REINFORCE, A2C and PPO
    optimise the expected return of a *stochastic* policy, which deliberately spreads
    probability over equally good villages; its argmax is a different policy (early in
    training it simply stays at the hub). They are evaluated by sampling, seeded per
    episode so results are reproducible.
    """
    return "greedy" if algo == "dqn" else "sample"


class TorchPolicy:
    """Adapts a trained TorchRL actor to the baseline Policy interface (obs -> action)."""

    def __init__(self, actor: nn.Module, name: str, sample: bool = False) -> None:
        self.actor = actor
        self.name = name
        self.sample = sample

    def reset(self, seed: int) -> None:
        if self.sample:
            torch.manual_seed(seed)  # same actions for the same episode, every time

    @staticmethod
    def as_tensordict(obs: np.ndarray) -> TensorDict:
        observation = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return TensorDict({"observation": observation, "action_mask": action_mask_from_obs(observation)},
                          batch_size=[1])

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> int:
        mode = ExplorationType.RANDOM if self.sample else ExplorationType.DETERMINISTIC
        with set_exploration_type(mode):
            return int(self.actor(self.as_tensordict(obs))["action"].item())


@lru_cache(maxsize=16)
def _do_nothing_reference(seeds: tuple[int, ...], scenario: str) -> pd.DataFrame:
    return evaluate(DoNothing(), seeds, scenario)


def evaluate_policy(policy, eval_sets: Iterable[str] = ("held_out",), limit: int | None = None) -> dict[str, Any]:
    """Evaluate ``policy`` on named seed sets; returns summary metrics and per-episode returns per set."""
    with torch.random.fork_rng():  # seeded sampling must not disturb anyone else's random state
        return _evaluate_sets(policy, eval_sets, limit)


def _evaluate_sets(policy, eval_sets: Iterable[str], limit: int | None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for set_name in eval_sets:
        seeds, scenario = EVAL_SETS[set_name]
        seeds = tuple(seeds[:limit]) if limit else tuple(seeds)
        env = AfyaSentinelEnv(scenario)
        episodes = pd.DataFrame([run_episode(policy, s, scenario, env) for s in seeds])
        episodes = add_infections_averted(episodes, _do_nothing_reference(seeds, scenario))
        summary = summarize(episodes).iloc[0].to_dict()
        results[set_name] = {
            **{k: (v.item() if hasattr(v, "item") else v) for k, v in summary.items()},
            "policy_mode": "sample" if getattr(policy, "sample", False) else "greedy",
            "seed_hash": seed_hash(seeds),
            "returns": episodes["return"].round(3).tolist(),
            "infections": episodes["infections"].round(2).tolist(),
            "invalid_actions": int(episodes["invalid"].sum()),
            "action_mix": {c: int(episodes[c].sum()) for c in ("n_travel", "n_screen", "n_prevent", "n_hub", "n_wait")},
        }
    return results


# ------------------------------------------------------------------------------ checkpoints


def save_run(cfg: RunConfig, modules: dict[str, nn.Module], stats: dict[str, Any], evaluation: dict[str, Any]) -> Path:
    """Write model.pt + config.json (model dir) and eval.json (log dir). Returns the log dir."""
    log_dir, model_dir = run_dirs(cfg)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save({name: m.state_dict() for name, m in modules.items()}, model_dir / "model.pt")
    meta = {
        "config": cfg.to_dict(),
        "env_version": ENV_VERSION,
        "versions": {
            "torch": torch.__version__,
            "torchrl": torchrl.__version__,
            "tensordict": tensordict.__version__,
            "python": platform.python_version(),
        },
        **stats,
    }
    (model_dir / "config.json").write_text(json.dumps(meta, indent=2))
    (log_dir / "eval.json").write_text(
        json.dumps({"algo": cfg.algo, "run_id": cfg.run_id, "seed": cfg.seed, "env_version": ENV_VERSION,
                    **stats, "eval": evaluation}, indent=2)
    )
    return log_dir


def load_policy(model_dir: str | Path, sample: bool | None = None) -> TorchPolicy:
    """Rebuild a saved actor from ``config.json`` + ``model.pt`` and wrap it as a policy.

    ``sample`` defaults to the algorithm's evaluation mode (see ``policy_mode``).
    """
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "config.json").read_text())
    cfg = RunConfig.from_dict(meta["config"])
    state = torch.load(model_dir / "model.pt", weights_only=True)
    if cfg.algo == "dqn":
        from training.dqn_training import build_qvalue_actor

        actor = build_qvalue_actor(cfg)
    else:
        from training.pg_training import build_actor

        actor = build_actor(cfg)
    actor.load_state_dict(state["actor"])
    actor.eval()
    if sample is None:
        sample = policy_mode(cfg.algo) == "sample"
    return TorchPolicy(actor, f"{cfg.algo}:{cfg.run_id}", sample=sample)


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
