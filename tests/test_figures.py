"""Figure and table generation from a fabricated run tree (fast: no training)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.best_configs import select_best
from analysis.load_runs import load_runs, model_dir
from analysis.make_figures import FIGURES, TABLES, build_all, changed_factors, episodes_to_converge
from environment.custom_env import ENV_VERSION
from training.common import ALGOS, default_config
from training.seeds import EVAL_SETS, HELD_OUT_SEEDS, seed_hash

ROOT = Path(__file__).resolve().parents[1]


def _fake_run(logs: Path, models: Path, algo: str, run_id: str, seed: int, sets, quality: float, rng) -> None:
    log_dir = logs / algo / run_id
    log_dir.mkdir(parents=True)
    episodes = np.arange(1, 401)
    frames = np.repeat(np.arange(1, 101) * 10_000, 4)
    returns = -1700 + quality * 1200 * (1 - np.exp(-episodes / 120)) + rng.normal(0, 60, episodes.size)
    pd.DataFrame({"frames": frames, "episode": episodes, "return": returns, "decisions": 40}).to_csv(
        log_dir / "train.csv", index=False)
    grid = np.arange(1, 201) * 5_000
    if algo == "dqn":
        updates = {"loss": np.exp(-grid / 3e5) + 0.05, "q_mean": -grid / 1e6, "grad_norm": 1.0,
                   "epsilon": np.maximum(0.05, 1 - grid / 3e5)}
    else:
        updates = {"loss_objective": 0.01, "entropy": 2.4 * np.exp(-grid / (6e5 * quality)) + 0.2, "loss_critic": 0.5,
                   "grad_norm": 0.3}
    pd.DataFrame({"frames": grid, "updates": np.arange(1, 201), **updates, "lr": 3e-4}).to_csv(
        log_dir / "updates.csv", index=False)
    evaluation = {}
    for set_name in sets:
        n = len(EVAL_SETS[set_name][0])
        rets = (-1600 + quality * 1000 + rng.normal(0, 200, n)).round(3).tolist()
        evaluation[set_name] = {
            "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets, ddof=1)),
            "infections_mean": 1200 - quality * 700, "infections_averted_mean": quality * 700, "outbreak_rate": 0.0,
            "stranded_rate": 0.1, "seed_hash": seed_hash(EVAL_SETS[set_name][0]),
            "policy_mode": "greedy" if algo == "dqn" else "sample", "invalid_actions": 0, "returns": rets,
            "infections": [1200 - quality * 700] * n,
            "action_mix": {"n_travel": 400, "n_screen": 600, "n_prevent": 300, "n_hub": 150, "n_wait": 50},
        }
    record = {"algo": algo, "run_id": run_id, "seed": seed, "env_version": ENV_VERSION, "frames": 1_000_000,
              "updates": 200, "episodes": 400, "train_seconds": 100.0, "fps": 10_000.0, "eval": evaluation}
    if algo != "dqn":
        record["argmax_held_out"] = {"return_mean": -800.0, "return_std": 200.0, "infections_averted_mean": 500.0,
                                     "stranded_rate": 0.1}
    (log_dir / "eval.json").write_text(json.dumps(record))
    directory = model_dir(models, algo, run_id)
    directory.mkdir(parents=True)
    cfg = default_config(algo, run_id=run_id, seed=seed)
    (directory / "config.json").write_text(json.dumps({"config": cfg.to_dict(), "env_version": ENV_VERSION}))


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("figures")
    logs, models, assets = root / "logs", root / "models", root / "assets"
    rng = np.random.default_rng(0)
    variants = {"dqn": ["baseline", "double_dqn", "target_soft"],
                "reinforce": ["baseline", "entropy_0.01", "entropy_0.05"],
                "a2c": ["baseline", "entropy_0", "entropy_0.05"], "ppo": ["baseline", "entropy_0", "entropy_0.05"]}
    for i, algo in enumerate(ALGOS):
        for j, run_id in enumerate(variants[algo]):
            _fake_run(logs, models, algo, run_id, 0, ["held_out"], 0.5 + 0.1 * i - 0.1 * j, rng)
        for seed in (0, 1, 2):
            _fake_run(logs, models, algo, f"final_baseline_seed{seed}", seed, list(EVAL_SETS), 0.5 + 0.1 * i, rng)
    (logs / "best_configs.json").write_text(json.dumps(dict.fromkeys(ALGOS, "baseline")))
    (assets / "tables").mkdir(parents=True)
    shutil.copy(ROOT / "assets" / "tables" / "baselines_episodes.csv", assets / "tables")
    return logs, models, assets


def test_every_figure_and_table_is_built_from_the_logs(tree):
    logs, models, assets = tree
    made, skipped = build_all(logs, models, assets)
    # the timeline replays a real model, which a fabricated tree does not have
    assert set(skipped) == {"episode_timeline.png"} and "model not available" in skipped["episode_timeline.png"]
    for name in FIGURES:
        if name != "episode_timeline.png":
            assert (assets / "figures" / name).stat().st_size > 10_000, name
    for name in TABLES:
        for ext in (".md", ".csv"):
            assert (assets / "tables" / f"{name}{ext}").exists(), name
    hyper = pd.read_csv(assets / "tables" / "hyperparams_dqn.csv")
    assert hyper["run"].iloc[0] == "★ baseline" and "double_dqn=True" in hyper["changed"].tolist()
    summary = pd.read_csv(assets / "tables" / "final_summary.csv")
    assert list(summary["algorithm"]) == ["DQN", "REINFORCE", "A2C", "PPO"] and (summary["seeds"] == 3).all()
    gen = pd.read_csv(assets / "tables" / "generalisation.csv")
    assert set(gen["set"]) == set(EVAL_SETS) and {"greedy", "do-nothing", "PPO"} <= set(gen["policy"])


def test_regenerating_gives_the_same_files(tree):
    logs, models, assets = tree
    build_all(logs, models, assets)
    first = sorted(p.name for p in (assets / "figures").iterdir())
    for p in (assets / "figures").iterdir():
        p.unlink()
    build_all(logs, models, assets)
    assert sorted(p.name for p in (assets / "figures").iterdir()) == first


def test_best_config_is_the_highest_held_out_return(tree):
    logs, models, _ = tree
    best = select_best(load_runs(logs, models))
    assert best == dict.fromkeys(ALGOS, "baseline")  # baseline has the highest quality in the fixture


def test_convergence_rule():
    rising = pd.Series(np.concatenate([np.zeros(100), np.linspace(0, 10, 100), np.full(200, 10.0)]))
    episode = episodes_to_converge(rising)
    assert 150 < episode < 260  # 90% of the gain is reached late in the ramp (rolling mean lags by the window)
    assert np.isnan(episodes_to_converge(pd.Series(np.zeros(300))))  # no improvement -> NaN
    assert np.isnan(episodes_to_converge(pd.Series(np.arange(60.0))))  # too short to judge


def test_tables_use_readable_numbers():
    from analysis.tables import readable_float

    assert readable_float(-1490.64) == "-1490.6"  # never 1.49e+03
    assert readable_float(171.0) == "171"
    assert readable_float(0.0003) == "0.0003"
    assert readable_float(7.162) == "7.16"
    assert readable_float(float("nan")) == "—"


def test_changed_factors_reads_the_sweep_definition():
    assert changed_factors("ppo", "clip_0.1") == "clip_epsilon=0.1"
    assert changed_factors("dqn", "baseline") == "baseline"
    assert changed_factors("a2c", "final_lr_7e-4_seed2") == "lr=0.0007"
    assert HELD_OUT_SEEDS  # seeds are importable for the fixture
