"""The committed result tables agree with the eval.json files they were built from."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.load_runs import load_runs
from training.common import ALGOS
from training.sweeps import SWEEPS

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "assets" / "tables"
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}

pytestmark = pytest.mark.skipif(not (TABLES / "final_summary.csv").exists(), reason="tables not generated")


def _mean_sd(text: str) -> tuple[float, float]:
    match = re.fullmatch(r"(-?[\d,.]+) ± ([\d,.]+)", str(text))
    assert match, text
    return float(match[1].replace(",", "")), float(match[2].replace(",", ""))


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return load_runs(ROOT / "logs", ROOT / "models")


@pytest.mark.parametrize("algo", ALGOS)
def test_hyperparameter_table_matches_each_runs_eval_json(algo):
    table = pd.read_csv(TABLES / f"hyperparams_{algo}.csv")
    assert sorted(table["run"].str.removeprefix("★ ")) == sorted(run_id for run_id, _ in SWEEPS[algo])
    assert len(table) >= 10
    for _, row in table.iterrows():
        run_id = row["run"].removeprefix("★ ")
        held_out = json.loads((ROOT / "logs" / algo / run_id / "eval.json").read_text())["eval"]["held_out"]
        mean, sd = _mean_sd(row["held_out_return"])
        assert mean == pytest.approx(held_out["return_mean"], abs=0.051), run_id
        assert sd == pytest.approx(held_out["return_std"], abs=0.051), run_id
        assert row["infections_averted"] == pytest.approx(held_out["infections_averted_mean"], abs=0.051), run_id
        assert row["stranded_rate"] == pytest.approx(held_out["stranded_rate"], abs=0.0051), run_id


def test_final_summary_matches_the_final_runs(runs):
    table = pd.read_csv(TABLES / "final_summary.csv").set_index("algorithm")
    final = runs[runs["run_id"].str.startswith("final_")]
    for algo in ALGOS:
        group = final[final["algo"] == algo]["held_out_return_mean"]
        row = table.loc[LABELS[algo]]
        assert row["seeds"] == len(group) == 3
        mean, sd = _mean_sd(row["held_out_return"])
        assert mean == pytest.approx(group.mean(), abs=0.051)
        assert sd == pytest.approx(group.std(ddof=1), abs=0.051)


def test_generalisation_table_matches_the_final_runs_and_baselines(runs):
    table = pd.read_csv(TABLES / "generalisation.csv")
    final = runs[runs["run_id"].str.startswith("final_")]
    for algo in ALGOS:
        group = final[final["algo"] == algo]
        for _, row in table[table["policy"] == LABELS[algo]].iterrows():
            assert row["return_mean"] == pytest.approx(group[f"{row['set']}_return_mean"].mean(), abs=1e-6)
            assert row["infections_averted_mean"] == pytest.approx(
                group[f"{row['set']}_infections_averted_mean"].mean(), abs=1e-6)
    assert set(table["set"]) == {"train", "held_out", "heavy_rain", "shuffled_hotspots"}
    # the held-out baseline rows are the committed baseline episodes
    episodes = pd.read_csv(TABLES / "baselines_episodes.csv")
    for policy, label in (("greedy", "greedy"), ("do_nothing", "do-nothing"), ("random", "random")):
        for scenario, set_name in (("default", "held_out"), ("heavy_rain", "heavy_rain")):
            expected = episodes[(episodes["policy"] == policy) & (episodes["scenario"] == scenario)]["return"].mean()
            got = table[(table["policy"] == label) & (table["set"] == set_name)]["return_mean"].item()
            assert got == pytest.approx(expected, abs=1e-6)
    assert np.isnan(table[table["policy"] == "greedy"]["training_seeds"]).all()
