"""The committed best models: they load, act validly, and reproduce their logged evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.load_runs import model_dir
from api.app import resolve_model_dir
from environment.custom_env import ENV_VERSION, AfyaSentinelEnv
from training.common import ALGOS, action_mask_from_obs, evaluate_policy, load_policy
from training.seeds import HELD_OUT_SEEDS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BEST = json.loads((MODELS / "best.json").read_text())


def test_best_json_names_a_model_for_every_algorithm():
    assert BEST["algo"] in ALGOS and set(BEST["models"]) == set(ALGOS)
    assert resolve_model_dir("best") == model_dir(MODELS, BEST["algo"], "best")


@pytest.mark.parametrize("algo", ALGOS)
def test_best_model_loads_and_acts_validly(algo):
    directory = resolve_model_dir(algo)
    assert directory == model_dir(MODELS, algo, "best")
    meta = json.loads((directory / "config.json").read_text())
    record = json.loads((directory / "eval.json").read_text())
    assert meta["env_version"] == ENV_VERSION == record["env_version"]
    assert meta["config"]["algo"] == algo and record["run_id"] == BEST["models"][algo]["run_id"]

    policy = load_policy(directory)
    env = AfyaSentinelEnv()
    obs, _ = env.reset(seed=HELD_OUT_SEEDS[0])
    with torch.random.fork_rng():
        policy.reset(HELD_OUT_SEEDS[0])
        for _ in range(20):
            action, thinking = policy.act_and_explain(obs)
            valid = action_mask_from_obs(torch.as_tensor(obs)).numpy()
            assert 0 <= action < 12 and valid[action], (algo, action)
            assert thinking["values"][action] is not None
            obs, _, terminated, truncated, info = env.step(action)
            assert not info["invalid"]
            if terminated or truncated:
                break


@pytest.mark.parametrize("algo", ALGOS)
def test_best_model_reproduces_its_logged_held_out_returns(algo):
    """Replaying the first held-out seeds gives the returns stored in the published eval.json."""
    directory = resolve_model_dir(algo)
    logged = json.loads((directory / "eval.json").read_text())["eval"]["held_out"]["returns"][:3]
    replayed = evaluate_policy(load_policy(directory), ["held_out"], limit=3)["held_out"]["returns"]
    np.testing.assert_allclose(replayed, logged, atol=0.01)
