"""Baseline policies (do-nothing, random, greedy) and the shared evaluation loop.

A policy maps an observation to an action and may keep per-episode state through
``reset(seed)``. The greedy baseline sees only the observation plus the public road map,
the same information the RL agents get. Beating greedy is the evidence that the problem
needs RL.

    uv run python -m analysis.baselines            # writes assets/tables/baselines.{csv,md}
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from analysis.tables import write_table
from environment.custom_env import (
    A_HUB,
    A_PREVENT,
    A_SCREEN,
    A_WAIT,
    DAYS_SINCE_SCREEN_SCALE,
    ENV_VERSION,
    HUB,
    N_ACTIONS,
    N_VILLAGES,
    OBS_FUEL,
    OBS_KITS,
    OBS_LOCATION,
    OBS_DAYS_SINCE_SCREEN,
    OBS_REPORTED,
    ROAD_KM,
    SCENARIOS,
    AfyaSentinelEnv,
    EnvParams,
)
from training.seeds import HELD_OUT_SEEDS, seed_hash

ROOT = Path(__file__).resolve().parents[1]


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def __call__(self, obs: np.ndarray) -> int: ...


class DoNothing:
    """Never intervene: the van waits at the hub all season."""

    name = "do_nothing"

    def reset(self, seed: int) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> int:
        return A_WAIT


class RandomPolicy:
    """Uniformly random actions (seeded per episode, so evaluation is reproducible)."""

    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng([seed, 2026])

    def __call__(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(N_ACTIONS))


class Greedy:
    """Chase the highest reported fever rate.

    Drive to the village with the highest CHW-reported fever rate that has not been
    screened in the last ``recent_days`` days and screen it. Go back to the hub to restock
    when out of kits, or when fuel would not cover the trip there and on to the hub with a
    safety margin. Never uses bed nets and ignores the weather.
    """

    name = "greedy"

    def __init__(self, params: EnvParams | None = None, recent_days: float = 2.0, fuel_margin: float = 1.3):
        self.params = params if params is not None else EnvParams()
        self.recent_days = recent_days
        self.fuel_margin = fuel_margin

    def reset(self, seed: int) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> int:
        p = self.params
        location = int(np.argmax(obs[OBS_LOCATION]))
        fuel = float(obs[OBS_FUEL]) * p.fuel_capacity_km
        kits = round(float(obs[OBS_KITS]) * p.kit_capacity)
        if kits == 0:
            return A_HUB
        days_since = obs[OBS_DAYS_SINCE_SCREEN] * DAYS_SINCE_SCREEN_SCALE
        score = np.where(days_since >= self.recent_days, obs[OBS_REPORTED], -1.0)
        target = int(np.argmax(score)) + 1
        if location == target:
            return A_SCREEN
        if fuel < (ROAD_KM[location, target] + ROAD_KM[target, HUB]) * self.fuel_margin:
            return A_HUB
        return target - 1


BASELINES: dict[str, type] = {"do_nothing": DoNothing, "random": RandomPolicy, "greedy": Greedy}


def run_episode(policy: Policy, seed: int, scenario: str = "default", env: AfyaSentinelEnv | None = None) -> dict[str, Any]:
    """Play one episode and return its metrics (episode infections include the settled rest)."""
    env = env if env is not None else AfyaSentinelEnv(scenario)
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    total_reward, decisions, done = 0.0, 0, False
    while not done:
        action = policy(obs)
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        decisions += 1
        done = terminated or truncated
    totals = info["totals"]
    return {
        "policy": policy.name,
        "scenario": scenario,
        "seed": seed,
        "return": total_reward,
        "decisions": decisions,
        "days": info["day"],
        "infections": info["episode_infections"],
        "severe": totals["severe"],
        "early_detected": totals["early_detected"],
        "symptomatic_detected": totals["symptomatic_detected"],
        "severe_referred": totals["severe_referred"],
        "km": totals["km"],
        "invalid": totals["invalid"],
        "nets_placed": totals["nets_placed"],
        "screens": totals["screens"],
        "final_prevalence": info["district_prevalence"],
        "termination": info["termination_reason"],
        "n_travel": int(action_counts[:N_VILLAGES].sum()),
        "n_screen": int(action_counts[A_SCREEN]),
        "n_prevent": int(action_counts[A_PREVENT]),
        "n_hub": int(action_counts[A_HUB]),
        "n_wait": int(action_counts[A_WAIT]),
    }


def evaluate(policy: Policy, seeds: Iterable[int], scenario: str = "default") -> pd.DataFrame:
    """Per-episode metrics for ``policy`` on each seed (one shared env instance)."""
    env = AfyaSentinelEnv(scenario)
    return pd.DataFrame([run_episode(policy, seed, scenario, env) for seed in seeds])


def add_infections_averted(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add ``infections_averted`` = reference (do-nothing) infections − infections, matched by seed."""
    ref = reference.set_index(["scenario", "seed"])["infections"].rename("reference_infections")
    out = df.join(ref, on=["scenario", "seed"])
    out["infections_averted"] = out["reference_infections"] - out["infections"]
    return out.drop(columns="reference_infections")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (policy, scenario): mean ± std return and the headline outcome metrics."""
    rows = []
    for (policy, scenario), g in df.groupby(["policy", "scenario"], sort=False):
        rows.append(
            {
                "policy": policy,
                "scenario": scenario,
                "episodes": len(g),
                "return_mean": g["return"].mean(),
                "return_std": g["return"].std(ddof=1),
                "infections_mean": g["infections"].mean(),
                "infections_averted_mean": g["infections_averted"].mean() if "infections_averted" in g else np.nan,
                "outbreak_rate": (g["termination"] == "outbreak").mean(),
                "stranded_rate": (g["termination"] == "stranded").mean(),
                "days_mean": g["days"].mean(),
                "decisions_mean": g["decisions"].mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate_baselines(seeds: Sequence[int], scenarios: Sequence[str]) -> pd.DataFrame:
    """Evaluate every baseline on every scenario; adds infections averted vs do-nothing."""
    frames = []
    for scenario in scenarios:
        per_policy = {name: evaluate(cls(), seeds, scenario) for name, cls in BASELINES.items()}
        reference = per_policy["do_nothing"]
        frames += [add_infections_averted(df, reference) for df in per_policy.values()]
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the baseline policies on the held-out seeds.")
    parser.add_argument("--scenarios", nargs="+", default=["default"], choices=sorted(SCENARIOS))
    parser.add_argument("--out", default=str(ROOT / "assets" / "tables" / "baselines"), help="output path stem")
    args = parser.parse_args(argv)

    episodes = evaluate_baselines(HELD_OUT_SEEDS, args.scenarios)
    summary = summarize(episodes)
    summary.insert(0, "env_version", ENV_VERSION)
    summary["seed_hash"] = seed_hash(HELD_OUT_SEEDS)
    csv_path, md_path = write_table(summary, args.out)
    episodes.to_csv(Path(args.out).with_name(Path(args.out).name + "_episodes.csv"), index=False)
    print(md_path.read_text())
    print(f"wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
