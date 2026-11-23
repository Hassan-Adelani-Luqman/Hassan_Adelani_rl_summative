"""Report figures and tables, regenerated from the training logs by one command.

    uv run main.py figures [--strict]

Reads finished runs from logs/ (+ their configs in models/) and the baseline tables, and
writes PNG figures to assets/figures/ and Markdown + CSV tables to assets/tables/.
A figure whose data does not exist yet (e.g. before the final multi-seed runs) is skipped
with a reason; ``--strict`` turns any skip into an error.

Conventions: each algorithm keeps one colour everywhere (DQN blue, REINFORCE orange, A2C
aqua, PPO yellow; a validated colour-blind-safe order). Baselines are neutral grey reference
lines. Every chart has one y-axis, and every figure has a table with the same numbers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from analysis.baselines import BASELINES, evaluate  # noqa: E402
from analysis.best_configs import select_best, sweep_runs  # noqa: E402
from analysis.compare import paired_bootstrap  # noqa: E402
from analysis.load_runs import load_runs, load_train, load_updates, model_dir  # noqa: E402
from analysis.tables import write_table  # noqa: E402
from environment.custom_env import VILLAGE_NAMES  # noqa: E402
from training.common import ALGO_FIELDS, ALGOS, PG_ALGOS, ROOT  # noqa: E402
from training.seeds import EVAL_SETS, HELD_OUT_SEEDS  # noqa: E402
from training.sweeps import SWEEPS  # noqa: E402

# ------------------------------------------------------------------------------ style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a85", "#e4e3df"
ALGO_COLOURS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "a2c": "#1baf7a", "ppo": "#eda100"}
ALGO_LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
BASELINE_STYLE = {"greedy": ("#52514e", "--", "greedy"), "do_nothing": ("#8b8a85", ":", "do-nothing"),
                  "random": ("#b5b4ae", "-.", "random")}
ACTION_GROUPS = (("n_travel", "travel", "#2a78d6"), ("n_screen", "screen", "#eb6834"),
                 ("n_prevent", "bed nets", "#1baf7a"), ("n_hub", "restock", "#eda100"), ("n_wait", "wait", "#e87ba4"))
SET_LABELS = {"train": "training districts", "held_out": "held-out districts", "heavy_rain": "heavy rain",
              "shuffled_hotspots": "shuffled hotspots"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
                                                    "#0d366b"])
WINDOW = 50  # episodes in the rolling mean (also the convergence rule's window)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
    "axes.axisbelow": True,
})

FIGURES = ("cumulative_reward.png", "cumulative_reward_overlay.png", "dqn_loss.png", "pg_entropy.png",
           "episodes_to_converge.png", "generalisation.png", "action_mix.png", "episode_timeline.png",
           "sweep_overview.png")
TABLES = ("hyperparams_dqn", "hyperparams_reinforce", "hyperparams_a2c", "hyperparams_ppo", "final_summary",
          "generalisation")


class Skip(Exception):
    """A figure or table cannot be made from the data available yet."""


# ------------------------------------------------------------------------------ data


@dataclass
class Results:
    runs: pd.DataFrame
    sweep: pd.DataFrame
    final: pd.DataFrame
    best: dict[str, str]
    baselines: pd.DataFrame  # per-episode baselines (policy, scenario, seed, ...)
    models_root: Path
    tables_dir: Path
    cache: dict[str, Any] = field(default_factory=dict)

    def runs_for(self, algo: str) -> tuple[list[Path], str]:
        """Log dirs for an algorithm's headline curve: its final seeds, else its best sweep run."""
        final = self.final[self.final["algo"] == algo]
        if not final.empty:
            return [Path(p) for p in final["log_dir"]], f"{len(final)} seeds"
        if algo in self.best:
            row = self.sweep[(self.sweep["algo"] == algo) & (self.sweep["run_id"] == self.best[algo])]
            return [Path(p) for p in row["log_dir"]], "1 seed"
        raise Skip(f"no finished runs for {algo}")

    def baseline_mean(self, policy: str, set_name: str = "held_out", metric: str = "return") -> float:
        df = self.baseline_set(set_name)
        return float(df[df["policy"] == policy][metric].mean())

    def baseline_set(self, set_name: str) -> pd.DataFrame:
        """Per-episode baseline results on a seed set (computed and cached for the training seeds)."""
        seeds, scenario = EVAL_SETS[set_name]
        df = self.baselines[(self.baselines["scenario"] == scenario) & self.baselines["seed"].isin(seeds)]
        if len(df) == len(BASELINES) * len(seeds):
            return df
        if set_name not in self.cache:
            frames = [evaluate(cls(), seeds, scenario).assign(policy=name) for name, cls in BASELINES.items()]
            out = pd.concat(frames, ignore_index=True)
            ref = out[out["policy"] == "do_nothing"].set_index("seed")["infections"]
            out["infections_averted"] = out["seed"].map(ref) - out["infections"]
            self.cache[set_name] = out
        return self.cache[set_name]


def load_results(logs_root: Path, models_root: Path, tables_dir: Path) -> Results:
    runs = load_runs(logs_root, models_root)
    if runs.empty:
        runs = pd.DataFrame(columns=["algo", "run_id", "log_dir"])
    final = runs[runs["run_id"].astype(str).str.startswith("final_")] if not runs.empty else runs
    sweep = sweep_runs(runs) if not runs.empty else runs
    best_file = logs_root / "best_configs.json"
    best = json.loads(best_file.read_text()) if best_file.exists() else select_best(runs)
    baselines_path = tables_dir / "baselines_episodes.csv"
    baselines = pd.read_csv(baselines_path) if baselines_path.exists() else pd.DataFrame(
        columns=["policy", "scenario", "seed", "return", "infections"])
    return Results(runs, sweep, final, best, baselines, models_root, tables_dir)


def changed_factors(algo: str, run_id: str) -> str:
    overrides = dict(SWEEPS[algo]).get(run_id.removeprefix("final_").rsplit("_seed", 1)[0], {})
    return ", ".join(f"{k}={v}" for k, v in overrides.items()) or "baseline"


def episodes_to_converge(returns: pd.Series, window: int = WINDOW, share: float = 0.9) -> float:
    """First episode where the rolling mean reaches initial + share·(final − initial); NaN if it never improves."""
    if len(returns) < 2 * window:
        return float("nan")
    rolling = returns.rolling(window).mean()
    initial, final = returns.iloc[:window].mean(), returns.iloc[-window:].mean()
    if final <= initial:
        return float("nan")
    hit = np.flatnonzero(rolling.to_numpy() >= initial + share * (final - initial))
    return float(hit[0] + 1) if hit.size else float("nan")


def rolling_curve(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Rolling mean indexed by frames (several episodes can share a frame count)."""
    smooth = df[column].rolling(window, min_periods=max(1, window // 5)).mean()
    return smooth.groupby(df["frames"]).mean()


def band(curves: list[pd.Series], points: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 1 s.d. of several curves on a common frame grid (s.d. is 0 for a single curve)."""
    lo = max(c.index.min() for c in curves)
    hi = min(c.index.max() for c in curves)
    grid = np.linspace(lo, hi, points)
    stack = np.vstack([np.interp(grid, c.index.to_numpy(float), c.to_numpy(float)) for c in curves])
    return grid, stack.mean(axis=0), stack.std(axis=0)


def thousands(ax, axis: str = "x") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if abs(v) >= 1e6 else f"{v / 1e3:g}k")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Nudge sorted label positions apart so direct labels never overlap."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def reference_lines(ax, results: Results, policies=("greedy", "do_nothing"), set_name="held_out", label=True) -> None:
    for policy in policies:
        colour, style, name = BASELINE_STYLE[policy]
        value = results.baseline_mean(policy, set_name)
        if np.isnan(value):
            continue
        ax.axhline(value, color=colour, linestyle=style, linewidth=1.2, zorder=1)
        if label:
            ax.annotate(f"{name} {value:.0f}", xy=(1.0, value), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)


def save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ figures


def fig_cumulative_reward(results: Results, out: Path) -> None:
    """2×2: episode return (cumulative reward) during training, per algorithm, with baselines."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey=True)
    for ax, algo in zip(axes.flat, ALGOS, strict=True):
        dirs, seeds = results.runs_for(algo)
        grid, mean, sd = band([rolling_curve(load_train(d), "return", WINDOW) for d in dirs])
        colour = ALGO_COLOURS[algo]
        ax.fill_between(grid, mean - sd, mean + sd, color=colour, alpha=0.22, linewidth=0)
        ax.plot(grid, mean, color=colour)
        reference_lines(ax, results)
        ax.set_title(f"{ALGO_LABELS[algo]} ({seeds}, best config: {results.best.get(algo, '?')})")
        thousands(ax)
    for ax in axes[1]:
        ax.set_xlabel("training frames (decisions)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"episode return ({WINDOW}-episode mean)")
    fig.suptitle("Cumulative reward per episode during training (band: ±1 s.d. across seeds)", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "cumulative_reward.png")


def fig_cumulative_overlay(results: Results, out: Path) -> None:
    """All four algorithms on one axis, directly labelled."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ends = []
    for algo in ALGOS:
        dirs, _ = results.runs_for(algo)
        grid, mean, sd = band([rolling_curve(load_train(d), "return", WINDOW) for d in dirs])
        ax.fill_between(grid, mean - sd, mean + sd, color=ALGO_COLOURS[algo], alpha=0.15, linewidth=0)
        ax.plot(grid, mean, color=ALGO_COLOURS[algo], label=ALGO_LABELS[algo])
        ends.append((mean[-1], grid[-1], algo))
    reference_lines(ax, results)
    lo, hi = ax.get_ylim()
    label_y = spread_labels([v for v, _, _ in ends], min_gap=0.045 * (hi - lo))
    for (_, x, algo), y in zip(ends, label_y, strict=True):
        ax.annotate(ALGO_LABELS[algo], xy=(x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK, annotation_clip=False)
    ax.set_xlabel("training frames (decisions)")
    ax.set_ylabel(f"episode return ({WINDOW}-episode mean)")
    ax.set_title("Training curves of the four algorithms (best configuration each)", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=4, fontsize=8.5, borderaxespad=0.2)
    thousands(ax)
    ax.margins(x=0.1)
    save(fig, out, "cumulative_reward_overlay.png")


def fig_dqn_loss(results: Results, out: Path) -> None:
    """DQN: TD loss, mean Q and ε on separate panels sharing the frame axis (no dual axes)."""
    dirs, seeds = results.runs_for("dqn")
    updates = [load_updates(d) for d in dirs]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={"height_ratios": [2, 1.3, 1]})
    for ax, column, ylabel in zip(axes, ("loss", "q_mean", "epsilon"), ("TD loss (smooth L1)", "mean Q of chosen action",
                                                                           "exploration ε"), strict=True):
        curves = []
        for u in updates:
            w = max(5, len(u) // 100)
            curves.append(u[column].rolling(w, min_periods=1).mean().groupby(u["frames"]).mean())
        grid, mean, sd = band(curves)
        ax.fill_between(grid, mean - sd, mean + sd, color=ALGO_COLOURS["dqn"], alpha=0.22, linewidth=0)
        ax.plot(grid, mean, color=ALGO_COLOURS["dqn"])
        ax.set_ylabel(ylabel)
        if column == "loss":
            ax.set_yscale("log")
            plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)
    axes[-1].set_xlabel("training frames")
    thousands(axes[-1])
    axes[0].set_title(f"DQN training diagnostics ({seeds}, config: {results.best.get('dqn', '?')}; Q in scaled-reward units)",
                      loc="left")
    fig.tight_layout()
    save(fig, out, "dqn_loss.png")


def fig_pg_entropy(results: Results, out: Path) -> None:
    """Policy entropy for REINFORCE, A2C and PPO, with each sweep's entropy-coefficient variants."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, algo in zip(axes, PG_ALGOS, strict=True):
        dirs, seeds = results.runs_for(algo)
        curves = []
        for d in dirs:
            u = load_updates(d)
            curves.append(u["entropy"].rolling(max(3, len(u) // 60), min_periods=1).mean().groupby(u["frames"]).mean())
        grid, mean, sd = band(curves)
        colour = ALGO_COLOURS[algo]
        ax.fill_between(grid, mean - sd, mean + sd, color=colour, alpha=0.22, linewidth=0)
        ax.plot(grid, mean, color=colour, label=f"best ({results.best.get(algo, '?')})")
        for run_id, style in (("entropy_0", (0, (4, 2))), ("entropy_0.05", (0, (1, 1.5))), ("entropy_0.01", (0, (6, 2, 1, 2)))):
            row = results.sweep[(results.sweep["algo"] == algo) & (results.sweep["run_id"] == run_id)]
            if row.empty or run_id == results.best.get(algo):
                continue
            u = load_updates(row.iloc[0]["log_dir"])
            if "entropy" not in u:  # older A2C logs omit entropy when its coefficient is 0
                continue
            curve = u["entropy"].rolling(max(3, len(u) // 60), min_periods=1).mean().groupby(u["frames"]).mean()
            ax.plot(curve.index, curve.to_numpy(), color=colour, linewidth=1.2, linestyle=style, label=run_id)
        ax.axhline(np.log(12), color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(f"{ALGO_LABELS[algo]} ({seeds})", loc="left")
        ax.set_xlabel("training frames")
        thousands(ax)
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("policy entropy (nats)")
    axes[0].annotate("uniform over 12 actions", xy=(0.0, np.log(12)), xycoords=("axes fraction", "data"),
                     xytext=(2, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "pg_entropy.png")


def convergence_table(results: Results) -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        dirs, _ = results.runs_for(algo)
        for d in dirs:
            train = load_train(d)
            ep = episodes_to_converge(train["return"])
            frames = float(train["frames"].iloc[int(ep) - 1]) if not np.isnan(ep) else float("nan")
            rows.append({"algo": algo, "run": Path(d).name, "episodes_to_converge": ep, "frames_to_converge": frames})
    return pd.DataFrame(rows)


def fig_episodes_to_converge(results: Results, out: Path) -> None:
    conv = convergence_table(results)
    if conv["episodes_to_converge"].isna().all():
        raise Skip("no run improved enough to measure convergence")
    stats = conv.groupby("algo", sort=False)["episodes_to_converge"].agg(["mean", "std", "count"]).reindex(ALGOS)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(ALGOS))
    ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0), color=[ALGO_COLOURS[a] for a in ALGOS], width=0.6,
           edgecolor=SURFACE, linewidth=2, capsize=4, error_kw={"ecolor": INK2, "elinewidth": 1})
    for i, (mean, sd) in enumerate(zip(stats["mean"], stats["std"].fillna(0), strict=True)):
        if not np.isnan(mean):  # above the error bar, never on it
            ax.annotate(f"{mean:,.0f}", xy=(i, mean + sd), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8.5, color=INK)
    ax.margins(y=0.12)
    ax.set_xticks(x, [ALGO_LABELS[a] for a in ALGOS])
    ax.set_ylabel("episodes to converge")
    ax.set_title("Episodes until the 50-episode mean reaches 90% of its final gain (mean ± s.d. over seeds)",
                 loc="left", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    save(fig, out, "episodes_to_converge.png")


def generalisation_table(results: Results) -> pd.DataFrame:
    if results.final.empty:
        raise Skip("needs the final multi-seed runs (evaluated on every seed set)")
    rows = []
    for algo in ALGOS:
        final = results.final[results.final["algo"] == algo]
        for set_name in EVAL_SETS:
            col = f"{set_name}_return_mean"
            if final.empty or col not in final:
                continue
            rows.append({"policy": ALGO_LABELS[algo], "set": set_name, "return_mean": final[col].mean(),
                         "return_sd_across_seeds": final[col].std(ddof=1) if len(final) > 1 else 0.0,
                         "infections_averted_mean": final[f"{set_name}_infections_averted_mean"].mean(),
                         "seeds": len(final)})
    for policy in ("greedy", "random", "do_nothing"):
        for set_name in EVAL_SETS:
            df = results.baseline_set(set_name)
            df = df[df["policy"] == policy]
            rows.append({"policy": BASELINE_STYLE[policy][2], "set": set_name, "return_mean": df["return"].mean(),
                         "return_sd_across_seeds": float("nan"),
                         "infections_averted_mean": df["infections_averted"].mean() if "infections_averted" in df else float("nan"),
                         "seeds": 0})
    return pd.DataFrame(rows)


def fig_generalisation(results: Results, out: Path) -> None:
    table = generalisation_table(results)
    sets = list(EVAL_SETS)
    policies = [ALGO_LABELS[a] for a in ALGOS] + ["greedy", "do-nothing"]
    colours = [ALGO_COLOURS[a] for a in ALGOS] + [BASELINE_STYLE["greedy"][0], "#c9c8c2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    width = 0.8 / len(policies)
    x = np.arange(len(sets))
    for ax, metric, ylabel in ((axes[0], "return_mean", "mean episode return"),
                               (axes[1], "infections_averted_mean", "infections averted vs do-nothing")):
        for i, (policy, colour) in enumerate(zip(policies, colours, strict=True)):
            if metric == "infections_averted_mean" and policy == "do-nothing":
                continue  # zero by definition
            values = [table[(table["policy"] == policy) & (table["set"] == s)][metric].mean() for s in sets]
            is_agent = policy in ALGO_LABELS.values()
            errors = ([table[(table["policy"] == policy) & (table["set"] == s)]["return_sd_across_seeds"].mean()
                       for s in sets] if metric == "return_mean" and is_agent else None)  # seed spread, agents only
            ax.bar(x + (i - (len(policies) - 1) / 2) * width, values, width, yerr=errors,
                   color=colour, edgecolor=SURFACE, linewidth=1.5, label=policy,
                   error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
        ax.set_xticks(x, [SET_LABELS[s].replace(" ", "\n", 1) for s in sets], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color="#c9c8c2", linewidth=1)
    axes[0].set_title("Return on each seed set (higher is better)", loc="left")
    axes[1].set_title("Infections averted vs doing nothing", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()  # the left panel has every policy
    axes[1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "generalisation.png")


def action_mix_table(results: Results) -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        final = results.final[results.final["algo"] == algo]
        dirs = [Path(p) for p in final["log_dir"]] if not final.empty else results.runs_for(algo)[0]
        totals = dict.fromkeys((c for c, _, _ in ACTION_GROUPS), 0)
        for d in dirs:
            mix = json.loads((d / "eval.json").read_text())["eval"]["held_out"]["action_mix"]
            for key in totals:
                totals[key] += mix[key]
        rows.append({"policy": ALGO_LABELS[algo], **totals})
    for policy in ("greedy", "do_nothing"):
        df = results.baseline_set("held_out")
        df = df[df["policy"] == policy]
        rows.append({"policy": BASELINE_STYLE[policy][2], **{c: int(df[c].sum()) for c, _, _ in ACTION_GROUPS}})
    table = pd.DataFrame(rows)
    counts = table[[c for c, _, _ in ACTION_GROUPS]]
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [f"share_{label.replace(' ', '_')}" for _, label, _ in ACTION_GROUPS]
    return pd.concat([table, shares], axis=1)


def fig_action_mix(results: Results, out: Path) -> None:
    table = action_mix_table(results)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for (_, label, colour), share_col in zip(ACTION_GROUPS, [c for c in table if c.startswith("share_")], strict=True):
        shares = table[share_col].to_numpy()
        ax.barh(y, shares, left=left, color=colour, edgecolor=SURFACE, linewidth=2, height=0.62, label=label)
        for yi, l0, s in zip(y, left, shares, strict=True):
            if s >= 0.07:
                ax.text(l0 + s / 2, yi, f"{s:.0%}", ha="center", va="center", fontsize=8, color=INK)
        left += shares
    ax.set_yticks(y, table["policy"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of decisions on the 50 held-out districts")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=5, fontsize=8.5)
    save(fig, out, "action_mix.png")


def best_overall(results: Results) -> tuple[str, Path]:
    """The algorithm with the best final (else sweep) held-out return, and one of its model dirs."""
    source = results.final if not results.final.empty else results.sweep
    if source.empty:
        raise Skip("no finished runs")
    row = source.sort_values("held_out_return_mean", ascending=False).iloc[0]
    return str(row["algo"]), model_dir(results.models_root, row["algo"], row["run_id"])


def fig_episode_timeline(results: Results, out: Path) -> None:
    """One held-out season: true prevalence per village (heatmap) with the van's visits, vs baselines."""
    import torch

    from analysis.baselines import DoNothing, Greedy
    from environment.rendering import EpisodeRunner
    from training.common import load_policy

    algo, directory = best_overall(results)
    if not (directory / "model.pt").exists():
        raise Skip(f"model not available locally: {directory}")
    seed = HELD_OUT_SEEDS[0]

    def play(policy) -> list[dict]:
        runner = EpisodeRunner(policy, seed)
        frames = [runner.reset()]
        with torch.random.fork_rng():
            while not runner.done:
                frames.append(runner.step())
        return frames

    agent = play(load_policy(directory))
    greedy, nothing = play(Greedy()), play(DoNothing())
    days = np.array([f["day"] for f in agent])
    prevalence = np.array([[v["true_prevalence"] for v in f["villages"]] for f in agent]).T
    fig, (ax_map, ax_line) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                                          gridspec_kw={"height_ratios": [2.2, 1]})
    edges = np.concatenate([[0.0], (days[1:] + days[:-1]) / 2, [days[-1] + 0.25]])
    mesh = ax_map.pcolormesh(edges, np.arange(len(VILLAGE_NAMES) + 1) - 0.5, prevalence, cmap=BLUES, vmin=0,
                             vmax=max(0.12, prevalence.max()), shading="flat")
    for f in agent[1:]:
        v = f["location"] - 1
        if f["events"]["screened"] is not None:
            ax_map.scatter(f["day"], v, marker="^", s=46, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        elif f["events"]["netted"] is not None:
            ax_map.scatter(f["day"], v, marker="s", s=40, color=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3)
        elif v >= 0 and len(f["route"]) > 1:
            ax_map.scatter(f["day"], v, marker="o", s=14, color=INK, zorder=3)
    ax_map.set_yticks(range(len(VILLAGE_NAMES)), VILLAGE_NAMES)
    ax_map.invert_yaxis()
    ax_map.grid(False)
    last = agent[-1]
    if last["termination"] != "time":
        ax_map.annotate(f"season ended day {last['day']:.1f}:\n{last['termination']}", xy=(last["day"], 3.5),
                        xytext=(10, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    fig.suptitle(f"One held-out season (district seed {seed}) with {ALGO_LABELS[algo]}: true prevalence and "
                 f"the van's work", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.01, fraction=0.035)
    cbar.set_label("true prevalence", color=INK2)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles = [plt.Line2D([], [], marker=m, linestyle="", color=c, markeredgecolor=e, markersize=s, label=l)
               for m, c, e, s, l in (("o", INK, INK, 4, "arrived"), ("^", INK, SURFACE, 7, "screened"),
                                     ("s", SURFACE, INK, 6.5, "bed nets"))]
    ax_map.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3, fontsize=8.5,
                  borderaxespad=0.2)
    for frames, colour, style, name in ((agent, ALGO_COLOURS[algo], "-", ALGO_LABELS[algo]),
                                        (greedy, BASELINE_STYLE["greedy"][0], "--", "greedy"),
                                        (nothing, BASELINE_STYLE["do_nothing"][0], ":", "do-nothing")):
        x = [f["day"] for f in frames]
        y = [f["district"]["true_prevalence"] for f in frames]
        ax_line.plot(x, y, color=colour, linestyle=style, label=name)
        ax_line.annotate(name, xy=(x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=8, color=INK)
    ax_line.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_line.set_ylabel("district prevalence")
    ax_line.set_xlabel("day of the rainy season")
    ax_line.legend(loc="upper left", fontsize=8, ncols=3)
    fig.tight_layout()
    save(fig, out, "episode_timeline.png")


def fig_sweep_overview(results: Results, out: Path) -> None:
    """Held-out return of every sweep run, per algorithm, against the baselines."""
    if results.sweep.empty:
        raise Skip("no sweep runs")
    # A dot plot zoomed to the data: bars from zero would hide the differences between runs.
    values = results.sweep["held_out_return_mean"]
    refs = [results.baseline_mean(p) for p in ("greedy", "do_nothing")]
    span = (min(values.min(), *refs), max(values.max(), *refs))
    pad = 0.06 * (span[1] - span[0] or 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    for ax, algo in zip(axes.flat, ALGOS, strict=True):
        group = results.sweep[results.sweep["algo"] == algo].sort_values("held_out_return_mean")
        if group.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(group))
        ax.errorbar(group["held_out_return_mean"], y, xerr=group["held_out_return_std"] / np.sqrt(50), fmt="o",
                    color=ALGO_COLOURS[algo], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xlim(span[0] - pad, span[1] + pad)
        ax.set_ylim(-0.7, len(group) - 0.3)
        labels = [f"{'★ ' if r == results.best.get(algo) else ''}{r}" for r in group["run_id"]]
        ax.set_yticks(y, labels, fontsize=8)
        for tick, run_id in zip(ax.get_yticklabels(), group["run_id"], strict=True):
            if run_id in ("baseline", results.best.get(algo)):
                tick.set_fontweight("bold")
        for policy in ("greedy", "do_nothing"):
            colour, style, name = BASELINE_STYLE[policy]
            value = results.baseline_mean(policy)
            ax.axvline(value, color=colour, linestyle=style, linewidth=1.2)
            ax.annotate(name, xy=(value, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="top")
        ax.set_title(f"{ALGO_LABELS[algo]} sweep ({len(group)} runs)", loc="left")
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("held-out mean return (± s.e. over 50 districts)")
    fig.tight_layout()
    save(fig, out, "sweep_overview.png")


# ------------------------------------------------------------------------------ tables


def hyperparameter_table(results: Results, algo: str) -> pd.DataFrame:
    group = results.sweep[results.sweep["algo"] == algo]
    if group.empty:
        raise Skip(f"no {algo} sweep runs")
    order = {run_id: i for i, (run_id, _) in enumerate(SWEEPS[algo])}
    rows = []
    for row in sorted(group.to_dict("records"), key=lambda r: order[r["run_id"]]):
        train = load_train(row["log_dir"])
        record = {"run": ("★ " if row["run_id"] == results.best.get(algo) else "") + row["run_id"],
                  "changed": changed_factors(algo, row["run_id"])}
        for name in ALGO_FIELDS[algo]:
            record[name] = row.get(f"cfg_{name}")
        record.update({
            "held_out_return": f"{row['held_out_return_mean']:.1f} ± {row['held_out_return_std']:.1f}",
            "infections_averted": round(row["held_out_infections_averted_mean"], 1),
            "stranded_rate": round(row.get("held_out_stranded_rate", float("nan")), 2),
            "episodes_to_converge": episodes_to_converge(train["return"]),
            "final_train_return": round(train["return"].iloc[-WINDOW:].mean(), 1),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def final_summary_table(results: Results) -> pd.DataFrame:
    if results.final.empty:
        raise Skip("needs the final multi-seed runs")
    greedy = results.baseline_set("held_out")
    greedy = greedy[greedy["policy"] == "greedy"].set_index("seed").loc[list(HELD_OUT_SEEDS)]
    conv = convergence_table(results)
    rows = []
    for algo in ALGOS:
        final = results.final[results.final["algo"] == algo]
        if final.empty:
            continue
        per_district = np.mean([json.loads((Path(d) / "eval.json").read_text())["eval"]["held_out"]["returns"]
                                for d in final["log_dir"]], axis=0)
        cmp = paired_bootstrap(per_district, greedy["return"].to_numpy())
        c = conv[conv["algo"] == algo]
        rows.append({
            "algorithm": ALGO_LABELS[algo], "config": results.best.get(algo, "?"), "seeds": len(final),
            "held_out_return": f"{final['held_out_return_mean'].mean():.1f} ± {final['held_out_return_mean'].std(ddof=1):.1f}",
            "infections_averted": round(final["held_out_infections_averted_mean"].mean(), 1),
            "vs_greedy_return": f"{cmp['mean_diff']:+.1f} [{cmp['ci_low']:+.1f}, {cmp['ci_high']:+.1f}]",
            "better_than_greedy_on": f"{cmp['share_of_seeds_better']:.0%}",
            "stranded_rate": round(final["held_out_stranded_rate"].mean(), 2),
            "episodes_to_converge": f"{c['episodes_to_converge'].mean():,.0f} ± {c['episodes_to_converge'].std(ddof=1):,.0f}",
            "argmax_return": round(final["argmax_return_mean"].mean(), 1) if "argmax_return_mean" in final else float("nan"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ driver

FIGURE_BUILDERS: dict[str, Callable[[Results, Path], None]] = {
    "cumulative_reward.png": fig_cumulative_reward,
    "cumulative_reward_overlay.png": fig_cumulative_overlay,
    "dqn_loss.png": fig_dqn_loss,
    "pg_entropy.png": fig_pg_entropy,
    "episodes_to_converge.png": fig_episodes_to_converge,
    "generalisation.png": fig_generalisation,
    "action_mix.png": fig_action_mix,
    "episode_timeline.png": fig_episode_timeline,
    "sweep_overview.png": fig_sweep_overview,
}


def build_all(logs_root: Path, models_root: Path, assets: Path) -> tuple[list[str], dict[str, str]]:
    figures_dir, tables_dir = assets / "figures", assets / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(logs_root, models_root, tables_dir)
    made, skipped = [], {}
    for name, builder in FIGURE_BUILDERS.items():
        try:
            builder(results, figures_dir)
            made.append(name)
        except Skip as why:
            skipped[name] = str(why)
            plt.close("all")
    table_builders = {f"hyperparams_{a}": (lambda a=a: hyperparameter_table(results, a)) for a in ALGOS}
    table_builders["final_summary"] = lambda: final_summary_table(results)
    table_builders["generalisation"] = lambda: generalisation_table(results)
    for name, builder in table_builders.items():
        try:
            write_table(builder(), tables_dir / name)
            made.append(f"{name}.md")
        except Skip as why:
            skipped[f"{name}.md"] = str(why)
    try:
        write_table(action_mix_table(results), tables_dir / "action_mix")
        write_table(convergence_table(results), tables_dir / "convergence")
    except Skip as why:
        skipped["action_mix.md / convergence.md"] = str(why)
    return made, skipped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main.py figures", description="Regenerate report figures and tables.")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument("--models-root", default=str(ROOT / "models"))
    parser.add_argument("--assets", default=str(ROOT / "assets"))
    parser.add_argument("--strict", action="store_true", help="fail if any figure or table had to be skipped")
    args = parser.parse_args(argv)
    made, skipped = build_all(Path(args.logs_root), Path(args.models_root), Path(args.assets))
    print(f"made {len(made)}: {', '.join(made) or '-'}")
    for name, why in skipped.items():
        print(f"skipped {name}: {why}")
    return 1 if (args.strict and skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
