"""Viewer bridge: the static district map, one JSON frame per decision, and the terminal log.

The Three.js viewer (``environment/viewer``) draws everything from these payloads:
- ``scene()``: the Bugesera map (locations, roads, lakes) and the constants the HUD needs.
- ``EpisodeRunner``: plays one episode with a policy and returns a frame per decision with the
  van's route, stock, weather, flooded roads, what the agent sees vs the hidden truth per
  village, events, the reward, and the agent's "thinking" (probabilities or Q-values over
  the 12 actions plus a one-line rationale).
- ``terminal_line``: the verbose step log printed while the viewer plays.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from environment.custom_env import (
    _KM_PER_DEG_LAT,
    _KM_PER_DEG_LON,
    A_HUB,
    A_PREVENT,
    A_SCREEN,
    A_WAIT,
    ACTION_NAMES,
    DAYS_SINCE_SCREEN_SCALE,
    HUB,
    LOCATION_LATLON,
    LOCATION_NAMES,
    LOCATION_POS_KM,
    N_VILLAGES,
    OBS_DAYS_SINCE_SCREEN,
    OBS_FORECAST,
    OBS_NET_DAYS_LEFT,
    OBS_POSITIVITY,
    OBS_RAIN,
    OBS_REPORTED,
    POPULATION,
    ROAD_KM,
    ROADS,
    WETLAND_VILLAGES,
    AfyaSentinelEnv,
    route,
)

FLOOD_RAIN = 0.55  # a flood-prone road shows as flooded at or above this rain index

# Water features drawn for context (not part of the simulation). Coordinates from Wikipedia;
# the river line is stylised.
LAKES = (
    {"name": "Lake Rweru", "lat": -2.3778, "lon": 30.3236, "radius_km": 4.5},
    {"name": "Lake Cyohoha North", "lat": -2.239, "lon": 30.112, "radius_km": 1.4},
)
RIVER = {"name": "Nyabarongo", "points_km": [(-14.0, -6.0), (-9.0, 2.0), (-5.0, 8.5), (2.0, 12.5), (10.0, 13.0),
                                             (18.0, 11.0), (26.0, 12.5)]}

FRAME_KEYS = (
    "type", "step", "tick", "day", "max_day", "action", "action_name", "origin", "location", "route", "ticks",
    "duration_s", "reward", "return", "stock", "weather", "flooded_roads", "villages", "district", "events",
    "thinking", "totals", "terminated", "truncated", "termination",
)
VILLAGE_KEYS = ("reported", "reported_count", "true_prevalence", "infected", "transmission", "net_days",
                "days_since_screen", "positivity", "hotspot")


def latlon_to_km(lat: float, lon: float) -> tuple[float, float]:
    """(x east, y north) km from Nyamata, the same projection as the environment's map."""
    return ((lon - LOCATION_LATLON[HUB, 1]) * _KM_PER_DEG_LON, (lat - LOCATION_LATLON[HUB, 0]) * _KM_PER_DEG_LAT)


def scene() -> dict[str, Any]:
    """Static map and constants for the viewer."""
    env = AfyaSentinelEnv()
    p = env.params
    locations = []
    for i, name in enumerate(LOCATION_NAMES):
        x, y = LOCATION_POS_KM[i]
        locations.append({
            "id": i, "name": name, "x": round(float(x), 3), "y": round(float(y), 3),
            "lat": float(LOCATION_LATLON[i, 0]), "lon": float(LOCATION_LATLON[i, 1]), "hub": i == HUB,
            "population": None if i == HUB else int(POPULATION[i - 1]),
            "wetland": i != HUB and (i - 1) in WETLAND_VILLAGES,
        })
    roads = [{"id": k, "a": a, "b": b, "flood_prone": flood, "km": round(float(ROAD_KM[a, b]), 2)}
             for k, (a, b, flood) in enumerate(ROADS)]
    lakes = []
    for lake in LAKES:
        x, y = latlon_to_km(lake["lat"], lake["lon"])
        lakes.append({"name": lake["name"], "x": round(float(x), 3), "y": round(float(y), 3), "radius": lake["radius_km"]})
    return {
        "district": "Bugesera District, Rwanda",
        "hub": HUB,
        "locations": locations,
        "roads": roads,
        "lakes": lakes,
        "river": {"name": RIVER["name"], "points": [list(pt) for pt in RIVER["points_km"]]},
        "actions": list(ACTION_NAMES),
        "constants": {
            "max_ticks": p.max_ticks, "ticks_per_day": p.ticks_per_day, "net_days": p.net_ticks / p.ticks_per_day,
            "fuel_capacity_km": p.fuel_capacity_km, "kit_capacity": p.kit_capacity, "net_capacity": p.net_capacity,
            "outbreak_threshold": p.outbreak_threshold, "flood_rain": FLOOD_RAIN,
        },
    }


def _level(value: float) -> str:
    return "very high" if value > 0.6 else "high" if value > 0.35 else "moderate" if value > 0.15 else "low"


def rationale(obs: np.ndarray, action: int, location: int, thinking: dict[str, Any] | None) -> str:
    """A template sentence explaining the decision from what the agent observed."""
    here = LOCATION_NAMES[location]
    reported = obs[OBS_REPORTED]
    days_since = obs[OBS_DAYS_SINCE_SCREEN] * DAYS_SINCE_SCREEN_SCALE
    net_days = obs[OBS_NET_DAYS_LEFT] * 10
    rain, forecast = float(obs[OBS_RAIN]), float(obs[OBS_FORECAST])
    if action < N_VILLAGES:
        name = LOCATION_NAMES[action + 1]
        nets = f", nets {net_days[action]:.0f} days left" if net_days[action] > 0 else ", no nets"
        text = (f"Heading to {name}: {_level(reported[action])} fever reports, last screened "
                f"{days_since[action]:.0f} days ago{nets}.")
    elif action == A_SCREEN:
        v = location - 1
        found = f" Last test positivity {obs[OBS_POSITIVITY][v] * 0.3:.0%}." if obs[OBS_POSITIVITY][v] > 0 else ""
        text = f"Screening {here}: {_level(reported[v])} fever reports, {days_since[v]:.0f} days since the last visit.{found}"
    elif action == A_PREVENT:
        text = (f"Bed nets for {here}: rain {rain:.0%}, 3-day forecast {forecast:.0%}. Ten days of protection "
                f"covers the surge that follows rain about 5 days later.")
    elif action == A_HUB:
        text = f"Back to Nyamata to restock fuel, test kits and nets."
    else:
        text = f"Holding at {here}."
    if thinking and thinking.get("values"):
        valid = [v for v in thinking["values"] if v is not None]
        if thinking["kind"] == "probability" and thinking["values"][action] is not None:
            text += f" (p = {thinking['values'][action]:.2f})"
        elif thinking["kind"] == "q_value" and len(valid) > 1:
            ranked = sorted(valid, reverse=True)
            text += f" (Q margin over next best {(ranked[0] - ranked[1]):.3f})"
    return text


class EpisodeRunner:
    """Plays one episode and produces viewer frames. Works with trained and baseline policies."""

    def __init__(self, policy, seed: int, scenario: str = "default", tick_seconds: float = 0.6) -> None:
        self.policy = policy
        self.seed = seed
        self.scenario = scenario
        self.tick_seconds = tick_seconds
        self.env = AfyaSentinelEnv(scenario)
        self.done = False
        self.step_index = 0
        self.total_reward = 0.0

    def reset(self) -> dict[str, Any]:
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.policy.reset(self.seed)
        self.done = False
        self.step_index = 0
        self.total_reward = 0.0
        return self._frame(None, None, 0.0, 0, False, False)

    def step(self, speed: float = 1.0) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        obs_before, location_before = self.obs, self.env.location
        if hasattr(self.policy, "act_and_explain"):
            action, thinking = self.policy.act_and_explain(obs_before)
        else:
            action, thinking = int(self.policy(obs_before)), None
        thinking = dict(thinking or {"kind": None, "values": None, "mask": None})
        thinking["chosen"] = action
        thinking["rationale"] = rationale(obs_before, action, location_before, thinking)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.step_index += 1
        self.done = terminated or truncated
        frame = self._frame(action, thinking, reward, self.info["ticks_elapsed"], terminated, truncated)
        frame["duration_s"] = round(self.tick_seconds * max(1, frame["ticks"]) / max(speed, 1e-3), 3)
        return frame

    def summary(self) -> dict[str, Any]:
        totals = self.info["totals"]
        return {
            "type": "end", "seed": self.seed, "scenario": self.scenario, "return": round(self.total_reward, 2),
            "infections": round(self.info["episode_infections"], 1), "severe": totals["severe"],
            "early_detected": totals["early_detected"], "symptomatic_detected": totals["symptomatic_detected"],
            "severe_referred": totals["severe_referred"], "screens": totals["screens"],
            "nets_placed": totals["nets_placed"], "km": round(totals["km"], 1), "days": self.info["day"],
            "termination": self.info["termination_reason"], "decisions": self.step_index,
        }

    def _frame(self, action, thinking, reward, ticks, terminated, truncated) -> dict[str, Any]:
        env, info, obs = self.env, self.info, self.obs
        p = env.params
        origin = info.get("origin", env.location)
        path = route(origin, env.location) if action is not None and origin != env.location else [env.location]
        if info.get("stranded"):
            dest = action + 1 if action < N_VILLAGES else HUB
            path = route(origin, dest)  # drove towards dest and ran dry on the way
        rain = env.rain_at(env.tick)
        compartments = info["compartments"]
        infected = compartments[1:4].sum(axis=0)
        net_days = env.net_ticks_left / p.ticks_per_day
        villages = []
        for v in range(N_VILLAGES):
            villages.append({
                "reported": round(float(obs[OBS_REPORTED][v]), 4),
                "reported_count": int(env.reported[v]),
                "true_prevalence": round(float(infected[v] / POPULATION[v]), 4),
                "infected": int(infected[v]),
                "transmission": round(float(info["transmission"][v]), 5),
                "net_days": round(float(net_days[v]), 2),
                "days_since_screen": round(float(env.days_since_screen[v]), 2),
                "positivity": round(float(env.last_positivity[v]), 4),
                "hotspot": bool(v in info["hotspots"]),
            })
        screened = env.location - 1 if action == A_SCREEN and not info.get("invalid") else None
        netted = env.location - 1 if action == A_PREVENT and not info.get("invalid") else None
        return {
            "type": "frame" if action is not None else "start",
            "step": self.step_index, "tick": env.tick, "day": env.tick / p.ticks_per_day,
            "max_day": p.max_ticks / p.ticks_per_day,
            "action": action, "action_name": ACTION_NAMES[action] if action is not None else None,
            "origin": origin, "location": env.location, "route": path, "ticks": int(ticks),
            "duration_s": 0.0, "reward": round(float(reward), 3), "return": round(self.total_reward, 3),
            "stock": {"fuel_km": round(env.fuel, 1), "fuel_frac": round(env.fuel / p.fuel_capacity_km, 4),
                      "kits": env.kits, "nets": env.nets},
            "weather": {"rain": round(rain, 3), "forecast": round(float(obs[OBS_FORECAST]), 3),
                        "rain_multiplier": round(env.rain_multiplier(env.tick), 3)},
            "flooded_roads": [k for k, (_, _, flood) in enumerate(ROADS) if flood and rain >= FLOOD_RAIN],
            "villages": villages,
            "district": {"true_prevalence": round(info["district_prevalence"], 4),
                         "reported_mean": round(float(np.mean(obs[OBS_REPORTED])), 4)},
            "events": {"early": info.get("early_detected", 0), "symptomatic": info.get("symptomatic_detected", 0),
                       "severe": info.get("severe_referred", 0), "new_infections": info.get("new_infections", 0),
                       "new_severe": info.get("new_severe", 0), "invalid": bool(info.get("invalid", False)),
                       "stranded": bool(info.get("stranded", False)), "screened": screened, "netted": netted},
            "thinking": thinking,
            "totals": {"infections": round(info["episode_infections"], 1), "screens": info["totals"]["screens"],
                       "nets_placed": info["totals"]["nets_placed"], "km": round(info["totals"]["km"], 1)},
            "terminated": bool(terminated), "truncated": bool(truncated),
            "termination": info.get("termination_reason"),
        }


# ------------------------------------------------------------------------------ terminal log

_COLOURS = {"travel": "36", "screen": "32", "prevent": "35", "hub": "33", "wait": "90", "bad": "31"}


def _paint(text: str, kind: str, colour: bool) -> str:
    return f"\033[{_COLOURS[kind]}m{text}\033[0m" if colour else text


def terminal_header(label: str, seed: int, scenario: str) -> str:
    return (f"\nAfya Sentinel | Bugesera District | agent {label} | district seed {seed} | scenario {scenario}\n"
            f"{'day':>5} | {'where':<8} | {'decision':<20} | {'found (early/symp/sev)':<22} | {'new inf':>7} | "
            f"{'reward':>8} | {'return':>8} | {'fuel':>6} {'kits':>4} {'nets':>4} | {'rain':>4}")


def terminal_line(frame: dict[str, Any], colour: bool = False) -> str:
    """One aligned line per decision for the terminal log."""
    action = frame["action"]
    kind = ("travel" if action < N_VILLAGES else {A_SCREEN: "screen", A_PREVENT: "prevent", A_HUB: "hub",
                                                    A_WAIT: "wait"}[action])
    ev = frame["events"]
    if ev["stranded"]:
        kind = "bad"
    decision = frame["action_name"] + (" (STRANDED)" if ev["stranded"] else " (invalid)" if ev["invalid"] else "")
    found = f"{ev['early']:>3} / {ev['symptomatic']:>3} / {ev['severe']:>2}" if action == A_SCREEN else ""
    where = LOCATION_NAMES[frame["origin"]]
    stock = frame["stock"]
    line = (f"{frame['day']:5.1f} | {where:<8} | {_paint(f'{decision:<20}', kind, colour)} | {found:<22} | "
            f"{ev['new_infections']:>7} | {frame['reward']:>8.1f} | {frame['return']:>8.1f} | "
            f"{stock['fuel_km']:>4.0f}km {stock['kits']:>4} {stock['nets']:>4} | {frame['weather']['rain']:>4.0%}")
    thinking = frame.get("thinking") or {}
    if thinking.get("rationale"):
        line += f"\n      {_paint('> ' + thinking['rationale'], 'wait', colour)}"
    return line


def terminal_summary(summary: dict[str, Any], reference: dict[str, dict[str, Any]] | None = None) -> str:
    lines = [f"  episode over ({summary['termination']}, day {summary['days']:.1f}): return {summary['return']:.1f}, "
             f"infections {summary['infections']:.0f}, screens {summary['screens']}, nets {summary['nets_placed']}, "
             f"{summary['km']:.0f} km"]
    for name, ref in (reference or {}).items():
        lines.append(f"  same district, {name}: return {ref['return']:.1f}, infections {ref['infections']:.0f}")
    return "\n".join(lines)


def reference_runs(seed: int, scenario: str, names: tuple[str, ...] = ("do_nothing", "greedy")) -> dict[str, dict]:
    """Baseline results on the same district, for the end-of-episode scoreboard."""
    from analysis.baselines import BASELINES, run_episode

    out = {}
    for name in names:
        r = run_episode(BASELINES[name](), seed, scenario)
        out[name] = {"return": round(r["return"], 2), "infections": round(r["infections"], 1)}
    return out


def isfinite_payload(payload: Any) -> bool:
    """True when every number in a payload is finite (JSON has no NaN/inf)."""
    if isinstance(payload, dict):
        return all(isfinite_payload(v) for v in payload.values())
    if isinstance(payload, list | tuple):
        return all(isfinite_payload(v) for v in payload)
    if isinstance(payload, float):
        return math.isfinite(payload)
    return True
