"""Training-stack tests: the TorchRL env, seed pool, sweep definitions, a micro-run of every
algorithm (save -> load -> same actions), and the resumable shard runner."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from environment.custom_env import ENV_VERSION, AfyaSentinelEnv
from training.cli import parse_overrides, train
from training.common import (
    ALGO_FIELDS,
    ALGOS,
    RunConfig,
    SeedPoolReset,
    TorchPolicy,
    default_config,
    evaluate_policy,
    load_policy,
    make_env,
    run_dirs,
)
from training.run_sweep import is_done
from training.seeds import HELD_OUT_SEEDS, TRAIN_SEED_POOL, seed_hash
from training.sweeps import SWEEPS, shard_configs

ROOT = Path(__file__).resolve().parents[1]


def test_torchrl_env_passes_spec_checks():
    env = make_env(seed=0, num_envs=2)
    check_env_specs(env)
    rollout = env.rollout(80, break_when_any_done=False)
    assert rollout["observation"].shape == (2, 80, 55)
    assert rollout["action"].dtype == torch.int64
    assert ((rollout["action"] >= 0) & (rollout["action"] < 12)).all()
    # episode_reward stays in env units while the agent sees the scaled reward
    done = rollout["next", "done"].squeeze(-1)
    assert done.any()
    env.close()


def test_training_districts_come_from_the_seed_pool_only():
    first, second = SeedPoolReset(AfyaSentinelEnv()), SeedPoolReset(AfyaSentinelEnv())
    first.reset(seed=7)
    second.reset(seed=7)
    seeds_a, seeds_b = [first.episode_seed], [second.episode_seed]
    for _ in range(50):
        first.reset()
        second.reset()
        seeds_a.append(first.episode_seed)
        seeds_b.append(second.episode_seed)
    assert seeds_a == seeds_b  # reproducible from the run seed
    assert len(set(seeds_a)) > 40  # varied
    assert all(s in TRAIN_SEED_POOL and s not in HELD_OUT_SEEDS for s in seeds_a)


@pytest.mark.parametrize("algo", ALGOS)
def test_every_sweep_has_a_baseline_and_ten_one_or_two_factor_variants(algo):
    runs = SWEEPS[algo]
    run_ids = [run_id for run_id, _ in runs]
    assert run_ids[0] == "baseline" and runs[0][1] == {}
    assert len(run_ids) == len(set(run_ids))
    assert len(runs) - 1 >= 10
    valid = {f.name for f in fields(RunConfig)}
    baseline = default_config(algo)
    for run_id, overrides in runs[1:]:
        assert 1 <= len(overrides) <= 2, run_id
        assert set(overrides) <= valid, run_id
        assert any(getattr(baseline, k) != v for k, v in overrides.items()), f"{run_id} changes nothing"
        assert set(overrides) <= set(ALGO_FIELDS[algo]) | {"target_update", "tau"}, run_id
    configs = shard_configs(algo, total_frames=123)
    assert {c.total_frames for c in configs} == {123}  # one shared budget per shard


MICRO = {
    "dqn": {"init_random_frames": 256, "steps_per_collect": 4, "updates_per_collect": 2},
    "reinforce": {"episodes_per_update": 4, "rollout_len": 32},
    "a2c": {"rollout_len": 16},
    "ppo": {"rollout_len": 32, "minibatch_size": 64, "ppo_epochs": 2},
}


@pytest.mark.parametrize("algo", ALGOS)
def test_micro_run_trains_saves_and_reloads(algo, tmp_path):
    torch.set_num_threads(1)
    cfg = default_config(algo, run_id="micro", total_frames=1024, num_envs=4, eval_limit=2,
                         logs_root=str(tmp_path / "logs"), models_root=str(tmp_path / "models"), **MICRO[algo])
    result = train(cfg)
    log_dir, model_dir = run_dirs(cfg)
    for path in (log_dir / "train.csv", log_dir / "updates.csv", log_dir / "eval.json",
                 model_dir / "model.pt", model_dir / "config.json"):
        assert path.exists(), path
    assert result["stats"]["frames"] >= cfg.total_frames
    assert result["stats"]["updates"] > 0

    saved = json.loads((log_dir / "eval.json").read_text())
    assert saved["env_version"] == ENV_VERSION
    assert saved["eval"]["held_out"]["seed_hash"] == seed_hash(HELD_OUT_SEEDS[:2])
    meta = json.loads((model_dir / "config.json").read_text())
    assert RunConfig.from_dict(meta["config"]) == cfg

    # the reloaded model acts exactly like the trained one, and re-evaluates identically
    trained, loaded = TorchPolicy(result["actor"], "trained"), load_policy(model_dir)
    for obs in np.random.default_rng(0).random((25, 55), dtype=np.float32):
        assert trained(obs) == loaded(obs)
    again = evaluate_policy(loaded, ["held_out"], limit=2)["held_out"]
    assert again["returns"] == saved["eval"]["held_out"]["returns"]


def test_resume_skips_runs_that_already_have_eval_json(tmp_path):
    configs = shard_configs("ppo", logs_root=str(tmp_path / "logs"))
    assert not any(is_done(c) for c in configs)
    finished = run_dirs(configs[3])[0]
    finished.mkdir(parents=True)
    (finished / "eval.json").write_text("{}")
    assert [is_done(c) for c in configs].count(True) == 1 and is_done(configs[3])


def test_set_overrides_are_typed():
    assert parse_overrides(["lr=1e-3", "double_dqn=true", "hidden=[64,64]", "target_update=soft"]) == {
        "lr": 1e-3, "double_dqn": True, "hidden": (64, 64), "target_update": "soft"}


@pytest.mark.parametrize("command", ["train", "sweep", "evaluate"])
def test_main_training_commands_have_help(command):
    result = subprocess.run([sys.executable, "main.py", command, "--help"], capture_output=True, text=True,
                            cwd=ROOT, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage: main.py " + command in result.stdout
