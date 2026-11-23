"""Policy-gradient training with TorchRL: REINFORCE, A2C and PPO.

    uv run python -m training.pg_training --algo ppo --frames 20000 --run-id smoke

All three share a ``ProbabilisticActor`` (categorical policy over the 12 actions) and a
``ValueOperator`` critic, and collect from the same vector environment as DQN.
- REINFORCE: complete episodes only. Discounted Monte Carlo returns are written to
  ``"advantage"`` (minus a learned V(s) when ``value_baseline`` is on) for ``ReinforceLoss``.
- A2C: one gradient step per rollout with ``A2CLoss`` and ``GAE`` advantages.
- PPO: ``ClipPPOLoss`` with ``GAE``, several epochs over minibatches drawn without
  replacement from a ``ReplayBuffer``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from environment.custom_env import N_ACTIONS, OBS_DIM
from training.common import (
    PG_ALGOS,
    RunConfig,
    RunLogger,
    Timer,
    TorchPolicy,
    action_spec,
    default_config,
    evaluate_policy,
    make_env,
    mlp,
    run_dirs,
    save_run,
)


def build_actor(cfg: RunConfig) -> ProbabilisticActor:
    logits = TensorDictModule(mlp(OBS_DIM, N_ACTIONS, cfg), in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(logits, in_keys=["logits"], distribution_class=Categorical, spec=action_spec(),
                              return_log_prob=True)


def build_critic(cfg: RunConfig) -> ValueOperator:
    return ValueOperator(mlp(OBS_DIM, 1, cfg), in_keys=["observation"])


def _optimise(loss_module, data: TensorDict, optimiser, cfg: RunConfig, extra: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """One gradient step on the sum of the module's loss terms (+ ``extra``); returns detached metrics."""
    out = loss_module(data)
    loss = sum(value for key, value in out.items() if key.startswith("loss_"))
    if extra is not None:
        loss = loss + extra
    optimiser.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
    optimiser.step()
    metrics = {key: value.detach().mean() for key, value in out.items() if value.numel() > 0}
    metrics["grad_norm"] = grad_norm.detach()
    return metrics


def _mean_metrics(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(torch.stack([row[key].float() for row in rows]).mean()) for key in keys}


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _train_actor_critic(cfg: RunConfig, env, actor, critic, logger: RunLogger) -> tuple[int, int]:
    """A2C and PPO: GAE advantages over [num_envs, rollout_len] batches."""
    collector = Collector(env, actor, frames_per_batch=cfg.num_envs * cfg.rollout_len, total_frames=cfg.total_frames)
    advantage = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic, average_gae=False)
    if cfg.algo == "a2c":
        loss_module = A2CLoss(actor, critic, entropy_coeff=cfg.entropy_coef, critic_coeff=cfg.critic_coef,
                              loss_critic_type="smooth_l1")
    else:
        loss_module = ClipPPOLoss(actor, critic, clip_epsilon=cfg.clip_epsilon, entropy_coeff=cfg.entropy_coef,
                                  critic_coeff=cfg.critic_coef, loss_critic_type="smooth_l1",
                                  normalize_advantage=cfg.normalize_advantage)
    optimiser = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    frames_per_batch = cfg.num_envs * cfg.rollout_len
    minibatch = min(cfg.minibatch_size, frames_per_batch)
    buffer = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch), sampler=SamplerWithoutReplacement(),
                          batch_size=minibatch)
    frames = updates = 0
    for batch in collector:
        frames += batch.numel()
        logger.episodes_from_batch(batch, frames)
        with torch.no_grad():
            advantage(batch)
        data = batch.reshape(-1)
        rows = []
        if cfg.algo == "a2c":
            if cfg.normalize_advantage:
                adv = data["advantage"]
                data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            rows.append(_optimise(loss_module, data, optimiser, cfg))
        else:
            for _ in range(cfg.ppo_epochs):
                buffer.empty()
                buffer.extend(data)
                for _ in range(frames_per_batch // minibatch):
                    rows.append(_optimise(loss_module, buffer.sample(), optimiser, cfg))
        updates += len(rows)
        logger.update(frames=frames, updates=updates, lr=cfg.lr, **_mean_metrics(rows))
    collector.shutdown()
    return frames, updates


def _train_reinforce(cfg: RunConfig, env, actor, critic, logger: RunLogger) -> tuple[int, int]:
    """REINFORCE on complete episodes; partial episodes carry over to the next batch."""
    collector = Collector(env, actor, frames_per_batch=cfg.num_envs * cfg.rollout_len, total_frames=cfg.total_frames)
    # ReinforceLoss always needs a critic. Without ``value_baseline`` the advantage is the raw
    # return, so the critic still fits V(s) on the side but never touches the actor's gradient.
    loss_module = ReinforceLoss(actor, critic, loss_critic_type="smooth_l1")
    optimiser = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    carry: list[TensorDict | None] = [None] * cfg.num_envs
    complete: list[TensorDict] = []
    frames = updates = 0
    for batch in collector:
        frames += batch.numel()
        logger.episodes_from_batch(batch, frames)
        for i in range(cfg.num_envs):
            row = batch[i] if carry[i] is None else torch.cat([carry[i], batch[i]], 0)
            start = 0
            for end in row["next", "done"].squeeze(-1).nonzero().squeeze(-1).tolist():
                complete.append(row[start : end + 1])
                start = end + 1
            carry[i] = row[start:] if start < row.shape[0] else None
        while len(complete) >= cfg.episodes_per_update:
            episodes, complete = complete[: cfg.episodes_per_update], complete[cfg.episodes_per_update :]
            for episode in episodes:
                episode["value_target"] = _discounted_returns(episode["next", "reward"], cfg.gamma)
            data = torch.cat(episodes, 0)
            returns = data["value_target"]
            if cfg.value_baseline:
                with torch.no_grad():
                    adv = returns - critic(data.select("observation"))["state_value"]
            else:
                adv = returns.clone()
            if cfg.normalize_returns:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            data["advantage"] = adv
            entropy = actor.get_dist(data).entropy().mean()
            metrics = _optimise(loss_module, data, optimiser, cfg, extra=-cfg.entropy_coef * entropy)
            metrics["entropy"] = entropy.detach()
            updates += 1
            logger.update(frames=frames, updates=updates, lr=cfg.lr, episodes_in_update=len(episodes),
                          **{k: float(v) for k, v in metrics.items()})
    collector.shutdown()
    return frames, updates


def train(cfg: RunConfig) -> dict[str, Any]:
    """Train one REINFORCE / A2C / PPO run, evaluate it on the held-out seeds and save everything."""
    if cfg.algo not in PG_ALGOS:
        raise ValueError(f"pg_training handles {PG_ALGOS}, not {cfg.algo!r}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    timer = Timer()
    env = make_env(cfg.seed, cfg.scenario, cfg.num_envs, cfg.reward_scale)
    actor, critic = build_actor(cfg), build_critic(cfg)
    log_dir, _ = run_dirs(cfg)
    logger = RunLogger(log_dir)
    if cfg.algo == "reinforce":
        frames, updates = _train_reinforce(cfg, env, actor, critic, logger)
    else:
        frames, updates = _train_actor_critic(cfg, env, actor, critic, logger)
    env.close()
    logger.close()

    train_seconds = timer.elapsed()
    evaluation = evaluate_policy(TorchPolicy(actor.eval(), f"{cfg.algo}:{cfg.run_id}"), cfg.eval_sets, cfg.eval_limit)
    stats = {"frames": frames, "updates": updates, "episodes": logger.episodes, "train_seconds": round(train_seconds, 1),
             "fps": round(frames / train_seconds, 1), "total_seconds": round(timer.elapsed(), 1)}
    modules = {"actor": actor, "critic": critic}
    save_run(cfg, modules, stats, evaluation)
    return {"stats": stats, "eval": evaluation}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train REINFORCE, A2C or PPO on Afya Sentinel.")
    parser.add_argument("--algo", choices=PG_ALGOS, required=True)
    parser.add_argument("--frames", type=int, default=None, help="total frames (default: config)")
    parser.add_argument("--run-id", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    overrides = {"run_id": args.run_id, "seed": args.seed}
    if args.frames:
        overrides["total_frames"] = args.frames
    result = train(default_config(args.algo, **overrides))
    held_out = result["eval"]["held_out"]
    print(f"{args.algo}:{args.run_id} {result['stats']} | held-out return {held_out['return_mean']:.1f} ± {held_out['return_std']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
