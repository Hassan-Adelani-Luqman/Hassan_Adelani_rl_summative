"""Hyperparameter sweeps: a baseline plus at least ten variants per algorithm.

Each variant changes one or two factors relative to that algorithm's baseline, so its
effect can be read directly against the baseline row of the report's tables. Every run
uses the same frame budget and is evaluated on the same held-out seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.common import ALGOS, ROOT, RunConfig, default_config

SWEEPS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "dqn": [
        ("baseline", {}),
        ("lr_1e-4", {"lr": 1e-4}),
        ("lr_2e-3", {"lr": 2e-3}),
        ("gamma_0.95", {"gamma": 0.95}),
        ("gamma_0.999", {"gamma": 0.999}),
        ("buffer_10k", {"buffer_size": 10_000}),
        ("batch_32", {"batch_size": 32}),
        ("batch_512", {"batch_size": 512}),
        ("eps_fast", {"eps_fraction": 0.1}),
        ("eps_slow_end_0.1", {"eps_fraction": 0.5, "eps_end": 0.1}),
        ("target_hard_2000", {"target_interval": 2000}),
        ("target_soft", {"target_update": "soft", "tau": 0.005}),
        ("double_dqn", {"double_dqn": True}),
    ],
    "reinforce": [
        ("baseline", {}),
        ("lr_3e-4", {"lr": 3e-4}),
        ("lr_3e-3", {"lr": 3e-3}),
        ("gamma_0.95", {"gamma": 0.95}),
        ("gamma_0.999", {"gamma": 0.999}),
        ("episodes_4", {"episodes_per_update": 4}),
        ("episodes_64", {"episodes_per_update": 64}),
        ("no_baseline", {"value_baseline": False}),
        ("entropy_0.01", {"entropy_coef": 0.01}),
        ("entropy_0.05", {"entropy_coef": 0.05}),
        ("no_return_norm", {"normalize_returns": False}),
        ("no_baseline_no_norm", {"value_baseline": False, "normalize_returns": False}),
    ],
    "a2c": [
        ("baseline", {}),
        ("lr_1e-4", {"lr": 1e-4}),
        ("lr_7e-4", {"lr": 7e-4}),
        ("gamma_0.95", {"gamma": 0.95}),
        ("gamma_0.999", {"gamma": 0.999}),
        ("no_adv_norm", {"normalize_advantage": False}),
        ("rollout_5", {"rollout_len": 5}),
        ("rollout_64", {"rollout_len": 64}),
        ("lambda_0.8", {"gae_lambda": 0.8}),
        ("lambda_1.0", {"gae_lambda": 1.0}),
        ("entropy_0", {"entropy_coef": 0.0}),
        ("entropy_0.05", {"entropy_coef": 0.05}),
        ("critic_coef_1.0", {"critic_coef": 1.0}),
        ("grad_clip_5", {"max_grad_norm": 5.0}),
    ],
    "ppo": [
        ("baseline", {}),
        ("lr_1e-4", {"lr": 1e-4}),
        ("lr_1e-3", {"lr": 1e-3}),
        ("clip_0.1", {"clip_epsilon": 0.1}),
        ("clip_0.3", {"clip_epsilon": 0.3}),
        ("lambda_0.9", {"gae_lambda": 0.9}),
        ("lambda_0.99", {"gae_lambda": 0.99}),
        ("epochs_2", {"ppo_epochs": 2}),
        ("epochs_10", {"ppo_epochs": 10}),
        ("minibatch_64", {"minibatch_size": 64}),
        ("entropy_0", {"entropy_coef": 0.0}),
        ("entropy_0.05", {"entropy_coef": 0.05}),
        ("rollout_256", {"rollout_len": 256}),
    ],
}

SHARDS = (*ALGOS, "test", "final")
TEST_FRAMES = 20_000
FINAL_SEEDS = (0, 1, 2)
BEST_CONFIGS_FILE = ROOT / "training" / "best_configs.json"  # {algo: run_id}: chosen from the sweeps, input to "final"


def sweep_configs(algo: str, **common: Any) -> list[RunConfig]:
    """All runs of one algorithm's sweep, with ``common`` overrides (e.g. frame budget) applied."""
    return [default_config(algo, run_id=run_id, **{**overrides, **common}) for run_id, overrides in SWEEPS[algo]]


def final_configs(best: dict[str, str] | None = None, seeds: tuple[int, ...] = FINAL_SEEDS, **common: Any) -> list[RunConfig]:
    """Best configuration of each algorithm, re-run with several seeds and evaluated on every seed set."""
    if best is None:
        best = json.loads(Path(BEST_CONFIGS_FILE).read_text())
    configs = []
    for algo, run_id in best.items():
        overrides = dict(SWEEPS[algo])[run_id]
        for seed in seeds:
            configs.append(default_config(algo, run_id=f"final_{run_id}_seed{seed}", seed=seed,
                                          eval_sets=("train", "held_out", "heavy_rain", "shuffled_hotspots"),
                                          **{**overrides, **common}))
    return configs


def shard_configs(shard: str, **common: Any) -> list[RunConfig]:
    """The runs in a shard: one algorithm's sweep, a short test of every baseline, or the final runs."""
    if shard in ALGOS:
        return sweep_configs(shard, **common)
    if shard == "test":
        return [default_config(algo, run_id="test", **{"total_frames": TEST_FRAMES, **common}) for algo in ALGOS]
    if shard == "final":
        return final_configs(**common)
    raise ValueError(f"unknown shard {shard!r}; choose from {SHARDS}")
