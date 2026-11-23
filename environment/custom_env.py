"""Afya Sentinel: a mobile malaria early-detection van in Bugesera District, Rwanda.

Bugesera (Eastern Province) is one of Rwanda's highest-burden malaria districts: low-lying,
with lakes, marshes and river wetlands. The agent dispatches one van from the district
hub at Nyamata to villages in eight of its sectors during the rainy season. It never
sees true infections, only noisy, one-day-late fever reports from community health
workers (CHWs), its own screening results and its stock levels. Rain drives
transmission about five days later and also floods the roads into wetland sectors, so
the villages that need the van most become the hardest to reach.

Key properties
- Partially observable: per-village S/E/I/Sev/R counts are hidden (they are in ``info``).
- Semi-MDP: one decision advances 1–4 half-day ticks. A travel action takes longer on
  flooded roads, and the disease keeps spreading while the van drives. The reward of a
  decision is the sum of its per-tick rewards. An episode is at most 60 ticks (30 days).
- No case farming: every case costs at least as much as any bonus it can earn, and an
  early termination charges the projected rest of the season, so it is never a shortcut.

The module-level constants (map, action ids, observation layout) are the single source
of truth for the baselines, the viewer and the report tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_VERSION = "1.0"  # frozen: sweeps and results depend on these dynamics

# ----------------------------------------------------------------------------------------
# Public district map: Bugesera District, Eastern Province, Rwanda.
# Location 0 is the hub (Nyamata, the district headquarters); locations 1..8 each stand
# for one village (umudugudu) in that sector, so populations are village-sized.
# Village-indexed arrays (length 8) use village index = location - 1.
#
# Coordinates (lat, lon): Nyamata, Mwogo, Rilima, Gashora, Mayange and Mareba from their
# Wikipedia articles; Ntarama from the Ntarama memorial; Rweru from Nemba (Rweru sector).
# Juru has no published point: it is placed east of Mwogo, as Mwogo's article describes,
# so its position is approximate. Roads and populations are stylised.
# ----------------------------------------------------------------------------------------
HUB = 0
N_VILLAGES = 8
N_LOCATIONS = N_VILLAGES + 1
LOCATION_NAMES = (
    "Nyamata",
    "Mwogo",
    "Juru",
    "Rilima",
    "Gashora",
    "Rweru",
    "Mayange",
    "Mareba",
    "Ntarama",
)
VILLAGE_NAMES = LOCATION_NAMES[1:]
LOCATION_LATLON = np.array(
    [
        [-2.1483, 30.0907],  # Nyamata (hub)
        [-2.0860, 30.1402],  # Mwogo
        [-2.1000, 30.2200],  # Juru (approximate)
        [-2.1597, 30.2258],  # Rilima
        [-2.2074, 30.2455],  # Gashora: eastern lakes (wetland)
        [-2.3525, 30.2175],  # Rweru: Lake Rweru, Burundi border (wetland)
        [-2.2065, 30.1414],  # Mayange
        [-2.2471, 30.0689],  # Mareba: near Lake Cyohoha North (wetland)
        [-2.1126, 30.0501],  # Ntarama: swamps on the Kigali side (wetland)
    ]
)
_KM_PER_DEG_LAT = 110.57
_KM_PER_DEG_LON = 111.32 * math.cos(math.radians(abs(LOCATION_LATLON[HUB, 0])))
# (x east, y north) in km from Nyamata
LOCATION_POS_KM = np.column_stack(
    (
        (LOCATION_LATLON[:, 1] - LOCATION_LATLON[HUB, 1]) * _KM_PER_DEG_LON,
        (LOCATION_LATLON[:, 0] - LOCATION_LATLON[HUB, 0]) * _KM_PER_DEG_LAT,
    )
)
POPULATION = np.array([420, 380, 300, 460, 350, 520, 280, 400], dtype=np.int64)
TOTAL_POPULATION = int(POPULATION.sum())
WETLAND_VILLAGES = np.array([3, 4, 6, 7])  # Gashora, Rweru, Mareba, Ntarama (village indices)
INLAND_VILLAGES = np.setdiff1d(np.arange(N_VILLAGES), WETLAND_VILLAGES)

# (location a, location b, flood-prone): roads into wetland sectors flood in heavy rain
ROADS: tuple[tuple[int, int, bool], ...] = (
    (0, 1, False),  # Nyamata – Mwogo
    (0, 3, False),  # Nyamata – Rilima
    (0, 6, False),  # Nyamata – Mayange
    (0, 7, True),  # Nyamata – Mareba
    (0, 8, True),  # Nyamata – Ntarama
    (1, 2, False),  # Mwogo – Juru
    (2, 3, False),  # Juru – Rilima
    (3, 4, True),  # Rilima – Gashora
    (6, 4, True),  # Mayange – Gashora
    (4, 5, True),  # Gashora – Rweru
    (6, 5, True),  # Mayange – Rweru
    (6, 7, True),  # Mayange – Mareba
    (8, 7, True),  # Ntarama – Mareba
)
ROAD_WINDING = 1.4  # road km per straight-line km on hilly murram roads


def _shortest_paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All-pairs shortest road paths (Floyd–Warshall): km, flood-prone fraction, next hop."""
    n = N_LOCATIONS
    km = np.full((n, n), np.inf)
    flood_km = np.zeros((n, n))
    next_hop = np.full((n, n), -1, dtype=np.int64)
    for i in range(n):
        km[i, i] = 0.0
        next_hop[i, i] = i
    for a, b, flood_prone in ROADS:
        length = ROAD_WINDING * float(np.linalg.norm(LOCATION_POS_KM[a] - LOCATION_POS_KM[b]))
        km[a, b] = km[b, a] = length
        flood_km[a, b] = flood_km[b, a] = length if flood_prone else 0.0
        next_hop[a, b], next_hop[b, a] = b, a
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if km[i, k] + km[k, j] < km[i, j] - 1e-9:
                    km[i, j] = km[i, k] + km[k, j]
                    flood_km[i, j] = flood_km[i, k] + flood_km[k, j]
                    next_hop[i, j] = next_hop[i, k]
    flood_frac = np.divide(flood_km, km, out=np.zeros_like(km), where=km > 0)
    return km, flood_frac, next_hop


ROAD_KM, ROUTE_FLOOD_FRAC, _NEXT_HOP = _shortest_paths()


def route(src: int, dst: int) -> list[int]:
    """Locations visited on the shortest road route from ``src`` to ``dst`` (inclusive)."""
    path = [src]
    while path[-1] != dst:
        path.append(int(_NEXT_HOP[path[-1], dst]))
    return path


def _spillover_matrix() -> np.ndarray:
    """Row-normalised mobility between villages joined by a direct road."""
    w = np.zeros((N_VILLAGES, N_VILLAGES))
    for a, b, _ in ROADS:
        if HUB not in (a, b):
            w[a - 1, b - 1] = w[b - 1, a - 1] = 1.0
    return w / w.sum(axis=1, keepdims=True)


SPILLOVER_W = _spillover_matrix()

# ----------------------------------------------------------------------------------------
# Hidden epidemic state: one (5, 8) integer array, rows = compartments, columns = villages.
# Flows per tick: S→E (infection), E→I (symptoms), I→Sev, I→R (recovery), Sev→R.
# ----------------------------------------------------------------------------------------
COMPARTMENTS = ("S", "E", "I", "Sev", "R")
S_, E_, I_, SEV_, R_ = range(5)
_FLOW_MATRIX = np.array(
    [
        # S→E  E→I  I→Sev  I→R  Sev→R
        [-1, 0, 0, 0, 0],  # S
        [1, -1, 0, 0, 0],  # E
        [0, 1, -1, -1, 0],  # I
        [0, 0, 1, 0, -1],  # Sev
        [0, 0, 0, 1, 1],  # R
    ],
    dtype=np.int64,
)

# ----------------------------------------------------------------------------------------
# Actions: Discrete(12)
# ----------------------------------------------------------------------------------------
A_SCREEN, A_PREVENT, A_HUB, A_WAIT = 8, 9, 10, 11
N_ACTIONS = 12
ACTION_NAMES = tuple(f"Travel to {name}" for name in VILLAGE_NAMES) + (
    "Screen",
    "Prevent (bed nets)",
    "Return to hub",
    "Wait",
)

# ----------------------------------------------------------------------------------------
# Observation: 55 float32 values in [0, 1]
# ----------------------------------------------------------------------------------------
OBS_LOCATION = slice(0, 9)  # one-hot: hub + 8 villages
OBS_FUEL = 9
OBS_KITS = 10
OBS_NETS = 11
OBS_DAY = 12
OBS_RAIN = 13
OBS_FORECAST = 14
OBS_REPORTED = slice(15, 23)  # CHW-reported fevers, 1-day lag, noisy
OBS_DAYS_SINCE_SCREEN = slice(23, 31)
OBS_POSITIVITY = slice(31, 39)  # last screening positivity
OBS_TRAVEL_TIME = slice(39, 47)  # ticks to reach each village under today's rain
OBS_NET_DAYS_LEFT = slice(47, 55)
OBS_DIM = 55
DAYS_SINCE_SCREEN_SCALE = 30.0


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights. Penalties are stored as positive magnitudes.

    Per decision:
        r = early_detected·E_found + symptomatic_detected·I_found + severe_referred·Sev_found
            − new_infection·new_infections − new_severe·new_severe_cases
            − per_km·km − invalid·[invalid action]
            − stranded·[stranded] − outbreak·[outbreak escaped] − (settled rest of season)

    No-farming invariant (tested): the largest bonus any single case can earn is at most
    the penalty that case costs, so no case is ever profitable and earlier detection
    always loses less.
    """

    early_detected: float = 1.0
    symptomatic_detected: float = 0.5
    severe_referred: float = 1.0
    new_infection: float = 1.0
    new_severe: float = 3.0
    per_km: float = 0.01
    invalid: float = 0.5
    stranded: float = 20.0
    outbreak: float = 30.0


@dataclass(frozen=True)
class EnvParams:
    """Simulation parameters. Time unit: one tick = half a day."""

    max_ticks: int = 60
    ticks_per_day: int = 2
    # van
    fuel_capacity_km: float = 150.0
    kit_capacity: int = 6  # screening batches
    net_capacity: int = 4  # bed-net bundles (one covers a village)
    speed_km_per_tick: float = 15.0  # effective, including murram roads and setting up on arrival
    max_travel_ticks: int = 4
    flood_slowdown: float = 1.5  # extra travel time on flooded km at rain = 1
    flood_fuel_extra: float = 0.3  # extra fuel on flooded km at rain = 1
    # rain
    rain_base: float = 0.25
    rain_amp: float = 0.45
    rain_period_ticks: float = 40.0
    rain_noise: float = 0.08
    storm_prob: float = 0.08
    storm_boost: float = 0.4
    forecast_ticks: int = 6  # 3-day forecast
    forecast_noise: float = 0.07
    rain_lag_ticks: tuple[int, int] = (8, 12)  # transmission follows rain 4–6 days earlier
    # disease (per tick)
    beta: float = 0.08  # calibrated: do-nothing infects ~40% by day 30; heavy rain escapes ~50%
    rain_mult_floor: float = 0.3
    rain_mult_gain: float = 1.7
    hotspot_mult: float = 1.6
    n_hotspots: int = 2
    hotspot_pool: str = "wetland"  # "wetland" | "all"
    spillover: float = 0.15
    importation: float = 3e-4
    early_infectiousness: float = 0.5
    severe_infectiousness: float = 0.5
    p_early_to_symptomatic: float = 1 / 6
    p_symptomatic_to_severe: float = 0.015
    p_symptomatic_recover: float = 0.07
    p_severe_recover: float = 0.06
    init_prevalence: tuple[float, float] = (0.005, 0.03)
    hotspot_init_prevalence: tuple[float, float] = (0.04, 0.09)
    outbreak_threshold: float = 0.25
    # interventions
    net_efficacy: float = 0.6
    net_ticks: int = 20  # 10 days of protection
    screen_coverage: float = 0.6
    sens_early: float = 0.85
    sens_symptomatic: float = 0.95
    severe_find_rate: float = 0.9
    # CHW reporting
    report_lag_ticks: int = 2
    report_completeness: tuple[float, float] = (0.5, 0.9)
    background_fever_rate: float = 0.004
    report_scale: float = 0.08  # reported fevers / population that maps to 1.0
    positivity_scale: float = 0.3  # screening positivity that maps to 1.0


SCENARIOS: dict[str, EnvParams] = {
    "default": EnvParams(),
    "heavy_rain": EnvParams(rain_base=0.45, storm_prob=0.15),
    "shuffled_hotspots": EnvParams(hotspot_pool="all"),
}


class AfyaSentinelEnv(gym.Env):
    """Gymnasium environment for the Afya Sentinel van-dispatch problem."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        scenario: str = "default",
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        params: EnvParams | None = None,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.scenario = scenario
        self.params = p = params if params is not None else SCENARIOS[scenario]
        self.rewards = reward_config if reward_config is not None else RewardConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        # constants derived from the parameters
        self._lag_pad = p.rain_lag_ticks[1]
        self._infectivity = np.array([p.early_infectiousness, 1.0, p.severe_infectiousness])
        self._mixing = (1.0 - p.spillover) * np.eye(N_VILLAGES) + p.spillover * SPILLOVER_W
        p_sev, p_rec = p.p_symptomatic_to_severe, p.p_symptomatic_recover
        self._p_sev_given_exit = p_sev / (p_sev + p_rec)
        self._probs = np.concatenate(  # S→E slot is filled each tick from the hazard
            (
                np.zeros(N_VILLAGES),
                np.full(N_VILLAGES, p.p_early_to_symptomatic),
                np.full(N_VILLAGES, p_sev + p_rec),
                np.full(N_VILLAGES, p.p_severe_recover),
            )
        )
        self._started = False

    # ------------------------------------------------------------------------------ API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        p, rng = self.params, self.np_random

        self.tick = 0
        self.location = HUB
        self.fuel = p.fuel_capacity_km
        self.kits = p.kit_capacity
        self.nets = p.net_capacity

        n_rain = self._lag_pad + p.max_ticks + p.forecast_ticks + 1
        self._rain = self._generate_rain(rng, n_rain)
        self._forecast_noise = rng.normal(0.0, p.forecast_noise, n_rain)
        self._precompute_weather()

        pool = WETLAND_VILLAGES if p.hotspot_pool == "wetland" else np.arange(N_VILLAGES)
        self.hotspots = np.sort(rng.choice(pool, size=p.n_hotspots, replace=False))
        is_hot = np.zeros(N_VILLAGES, dtype=bool)
        is_hot[self.hotspots] = True
        self._beta_hot = p.beta * np.where(is_hot, p.hotspot_mult, 1.0)

        prevalence = rng.uniform(*p.init_prevalence, N_VILLAGES)
        prevalence[is_hot] = rng.uniform(*p.hotspot_init_prevalence, int(is_hot.sum()))
        infected = rng.binomial(POPULATION, prevalence)
        early = rng.binomial(infected, 0.5)
        severe = rng.binomial(infected - early, 0.08)
        self.X = np.zeros((len(COMPARTMENTS), N_VILLAGES), dtype=np.int64)
        self.X[S_] = POPULATION - infected
        self.X[E_] = early
        self.X[I_] = infected - early - severe
        self.X[SEV_] = severe

        self._net_until = np.zeros(N_VILLAGES, dtype=np.int64)  # nets protect while tick < this
        days_ago = rng.uniform(3.0, 20.0, N_VILLAGES)
        self._last_screen_tick = -days_ago * p.ticks_per_day
        self.last_positivity = np.zeros(N_VILLAGES)
        self._completeness = rng.uniform(*p.report_completeness, N_VILLAGES)
        self._I_history = [self.X[I_].copy()]
        self._refresh_reports()
        self._last_hazard = self._hazard(self.tick, self.X, self._net_until)

        self.totals = {
            "infections": 0,
            "severe": 0,
            "early_detected": 0,
            "symptomatic_detected": 0,
            "severe_referred": 0,
            "km": 0.0,
            "invalid": 0,
            "nets_placed": 0,
            "screens": 0,
            "settled_infections": 0.0,
        }
        self._done = False
        self._started = True
        self._last_action: int | None = None
        self._last_reward = 0.0
        return self._observe(), self._info(None)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._started or self._done:
            raise RuntimeError("call reset() before step() (the episode has ended)")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"invalid action {action}")
        p, rw = self.params, self.rewards

        found = (0, 0, 0)  # early, symptomatic, severe
        km = 0.0
        invalid = stranded = False
        duration = 1
        origin = self.location

        if action < N_VILLAGES or action == A_HUB:
            dest = action + 1 if action < N_VILLAGES else HUB
            if dest == origin:
                if action == A_HUB:
                    self._restock()  # already at the hub: restocking takes one tick
                else:
                    invalid = True  # travelling to where the van already is
            else:
                duration = int(self._travel_table[self.tick, origin, dest])
                needed = self.fuel_needed(origin, dest, self.rain_at(self.tick))
                if needed > self.fuel + 1e-9:
                    stranded = True  # runs dry on the road; the trip time still passes
                    km = float(ROAD_KM[origin, dest]) * self.fuel / needed
                    self.fuel = 0.0
                else:
                    km = float(ROAD_KM[origin, dest])
                    self.fuel -= needed
                    self.location = dest
                    if dest == HUB:
                        self._restock()
        elif action == A_SCREEN:
            if origin == HUB or self.kits <= 0:
                invalid = True
            else:
                found = self._screen(origin - 1)
        elif action == A_PREVENT:
            if origin == HUB or self.nets <= 0:
                invalid = True
            else:
                self.nets -= 1
                self._net_until[origin - 1] = self.tick + p.net_ticks
                self.totals["nets_placed"] += 1
        # A_WAIT: hold position for one tick

        action_reward = (
            rw.early_detected * found[0]
            + rw.symptomatic_detected * found[1]
            + rw.severe_referred * found[2]
            - rw.per_km * km
            - rw.invalid * invalid
        )

        duration = min(duration, p.max_ticks - self.tick)
        tick_rewards: list[float] = []
        new_infections = new_severe = 0
        outbreak = False
        for k in range(duration):
            n_inf, n_sev = self._advance_tick()
            new_infections += n_inf
            new_severe += n_sev
            tick_rewards.append(
                -rw.new_infection * n_inf - rw.new_severe * n_sev + (action_reward if k == 0 else 0.0)
            )
            if self.district_prevalence() > p.outbreak_threshold:
                outbreak = True
                break

        terminated = stranded or outbreak
        truncated = (not terminated) and self.tick >= p.max_ticks
        terminal_penalty = settled_penalty = settled_infections = 0.0
        if terminated:
            terminal_penalty = -(rw.stranded * stranded + rw.outbreak * outbreak)
            settled_infections, settled_severe = self._project_rest_of_season()
            settled_penalty = -(rw.new_infection * settled_infections + rw.new_severe * settled_severe)
        reward = float(sum(tick_rewards) + terminal_penalty + settled_penalty)

        totals = self.totals
        totals["infections"] += new_infections
        totals["severe"] += new_severe
        totals["early_detected"] += found[0]
        totals["symptomatic_detected"] += found[1]
        totals["severe_referred"] += found[2]
        totals["km"] += km
        totals["invalid"] += int(invalid)
        totals["settled_infections"] += settled_infections
        self._done = terminated or truncated
        self._last_action = action
        self._last_reward = reward

        reason = "outbreak" if outbreak else "stranded" if stranded else "time" if truncated else None
        step_info = {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "origin": origin,
            "ticks_elapsed": len(tick_rewards),
            "tick_rewards": tick_rewards,
            "terminal_penalty": terminal_penalty,
            "settled_penalty": settled_penalty,
            "settled_infections": settled_infections,
            "new_infections": new_infections,
            "new_severe": new_severe,
            "early_detected": found[0],
            "symptomatic_detected": found[1],
            "severe_referred": found[2],
            "km": km,
            "invalid": invalid,
            "stranded": stranded,
            "termination_reason": reason,
        }
        return self._observe(), reward, terminated, truncated, self._info(step_info)

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        p = self.params
        last = "—" if self._last_action is None else ACTION_NAMES[self._last_action]
        lines = [
            f"Day {self.tick / p.ticks_per_day:4.1f}  at {LOCATION_NAMES[self.location]:<8}"
            f"  fuel {self.fuel:5.1f} km  kits {self.kits}  nets {self.nets}"
            f"  rain {self.rain_at(self.tick):.2f}  last: {last} ({self._last_reward:+.1f})",
            f"  {'village':<8} {'reported':>8} {'true prev':>9} {'nets(d)':>7} {'screened(d ago)':>15}",
        ]
        prev = self.village_prevalence()
        net_days = self.net_ticks_left / p.ticks_per_day
        since = self.days_since_screen
        for v, name in enumerate(VILLAGE_NAMES):
            hot = "*" if v in self.hotspots else " "
            lines.append(
                f" {hot}{name:<8} {self.reported[v]:>8d} {prev[v]:>9.1%} {net_days[v]:>7.1f} {since[v]:>15.1f}"
            )
        lines.append(f"  district prevalence {self.district_prevalence():.1%}  (* = hidden hotspot)")
        return "\n".join(lines)

    # ------------------------------------------------------------- state views and helpers
    @property
    def S(self) -> np.ndarray:
        return self.X[S_]

    @property
    def E(self) -> np.ndarray:
        return self.X[E_]

    @property
    def I(self) -> np.ndarray:  # noqa: E743 (epidemiological notation)
        return self.X[I_]

    @property
    def Sev(self) -> np.ndarray:
        return self.X[SEV_]

    @property
    def R(self) -> np.ndarray:
        return self.X[R_]

    @property
    def net_ticks_left(self) -> np.ndarray:
        return np.maximum(self._net_until - self.tick, 0)

    @property
    def days_since_screen(self) -> np.ndarray:
        return (self.tick - self._last_screen_tick) / self.params.ticks_per_day

    def travel_ticks(self, src: int, dst: int, rain: float) -> int:
        """Half-day ticks to drive from ``src`` to ``dst``; flooded road km are slower."""
        p = self.params
        if src == dst:
            return 0
        effective_km = ROAD_KM[src, dst] * (1.0 + p.flood_slowdown * rain * ROUTE_FLOOD_FRAC[src, dst])
        return int(min(p.max_travel_ticks, max(1, math.ceil(effective_km / p.speed_km_per_tick - 1e-9))))

    def fuel_needed(self, src: int, dst: int, rain: float) -> float:
        """Fuel (km-equivalent) for the trip; flooded road km burn more."""
        p = self.params
        return float(ROAD_KM[src, dst] * (1.0 + p.flood_fuel_extra * rain * ROUTE_FLOOD_FRAC[src, dst]))

    def rain_at(self, tick: int) -> float:
        """Rainfall index in [0, 1] at ``tick``."""
        return float(self._rain[tick + self._lag_pad])

    def rain_multiplier(self, tick: int) -> float:
        """Transmission multiplier driven by the mean rain 4–6 days (8–12 ticks) earlier."""
        return float(self._mult[tick])

    def forecast_at(self, tick: int) -> float:
        """Noisy 3-day rain forecast as observed at ``tick`` (mean rain over the next 6 ticks)."""
        return float(self._forecast[tick])

    def infection_hazard(self, with_nets: bool = True) -> np.ndarray:
        """Per-village infection hazard at the current tick (hidden; for tests and the viewer)."""
        return self._hazard(self.tick, self.X, self._net_until if with_nets else None)

    def village_prevalence(self) -> np.ndarray:
        """True share of each village currently infected (E + I + Sev)."""
        return self.X[E_ : SEV_ + 1].sum(axis=0) / POPULATION

    def district_prevalence(self) -> float:
        """True share of the district currently infected (E + I + Sev)."""
        return float(self.X[E_ : SEV_ + 1].sum() / TOTAL_POPULATION)

    # ---------------------------------------------------------------------- internals
    def _generate_rain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        p = self.params
        t = np.arange(n) - self._lag_pad
        phase = rng.uniform(0.0, 2.0 * math.pi)
        seasonal = p.rain_base + p.rain_amp * 0.5 * (1.0 + np.sin(2.0 * math.pi * t / p.rain_period_ticks + phase))
        storms = np.zeros(n)
        starts = np.flatnonzero(rng.random(n) < p.storm_prob)
        durations = rng.integers(2, 5, size=starts.size)
        for start, dur in zip(starts, durations, strict=True):
            storms[start : start + dur] += p.storm_boost
        noise = rng.normal(0.0, p.rain_noise, n)
        return np.clip(seasonal + storms + noise, 0.0, 1.0)

    def _precompute_weather(self) -> None:
        """Per-tick tables that depend only on the rain series (call again if it changes)."""
        p = self.params
        idx = np.arange(p.max_ticks + 1) + self._lag_pad
        csum = np.concatenate(([0.0], np.cumsum(self._rain)))
        lo, hi = p.rain_lag_ticks
        lagged_mean = (csum[idx - lo + 1] - csum[idx - hi]) / (hi - lo + 1)
        self._mult = p.rain_mult_floor + p.rain_mult_gain * lagged_mean
        ahead_mean = (csum[idx + 1 + p.forecast_ticks] - csum[idx + 1]) / p.forecast_ticks
        self._forecast = np.clip(ahead_mean + self._forecast_noise[idx], 0.0, 1.0)
        rain = self._rain[idx]
        effective_km = ROAD_KM[None] * (1.0 + p.flood_slowdown * rain[:, None, None] * ROUTE_FLOOD_FRAC[None])
        table = np.clip(np.ceil(effective_km / p.speed_km_per_tick - 1e-9), 1, p.max_travel_ticks)
        table[:, np.arange(N_LOCATIONS), np.arange(N_LOCATIONS)] = 0
        self._travel_table = table.astype(np.int64)  # [tick, src, dst]

    def _hazard(self, tick: int, X: np.ndarray, net_until: np.ndarray | None) -> np.ndarray:
        p = self.params
        infectious = (self._infectivity @ X[E_ : SEV_ + 1]) / POPULATION
        hazard = self._mult[tick] * (self._beta_hot * (self._mixing @ infectious) + p.importation)
        if net_until is not None:
            hazard = np.where(net_until > tick, hazard * (1.0 - p.net_efficacy), hazard)
        return hazard

    def _advance_tick(self) -> tuple[int, int]:
        p, rng, X = self.params, self.np_random, self.X
        hazard = self._hazard(self.tick, X, self._net_until)
        self._probs[:N_VILLAGES] = -np.expm1(-hazard)
        # One batched draw for S→E, E→I, I→exit, Sev→R; then split the symptomatic exits
        # between the two competing outcomes (severe vs recovered).
        draws = rng.binomial(X[:4].reshape(-1), self._probs).reshape(4, N_VILLAGES)
        i_to_sev = rng.binomial(draws[2], self._p_sev_given_exit)
        flows = np.stack((draws[0], draws[1], i_to_sev, draws[2] - i_to_sev, draws[3]))
        X += _FLOW_MATRIX @ flows
        self._last_hazard = hazard

        self.tick += 1
        self._I_history.append(X[I_].copy())
        if self.tick % p.ticks_per_day == 0:
            self._refresh_reports()
        return int(flows[0].sum()), int(flows[2].sum())

    def _refresh_reports(self) -> None:
        """Daily CHW fever reports: lagged symptomatic cases × completeness + background fevers."""
        p = self.params
        lagged = self._I_history[max(0, len(self._I_history) - 1 - p.report_lag_ticks)]
        rate = self._completeness * lagged + p.background_fever_rate * POPULATION
        self.reported = self.np_random.poisson(rate)

    def _screen(self, v: int) -> tuple[int, int, int]:
        p, rng, X = self.params, self.np_random, self.X
        tested = max(1, round(p.screen_coverage * POPULATION[v]))
        found_e = int(rng.binomial(X[E_, v], p.screen_coverage * p.sens_early))
        found_i = int(rng.binomial(X[I_, v], p.screen_coverage * p.sens_symptomatic))
        found_sev = int(rng.binomial(X[SEV_, v], p.severe_find_rate))
        X[E_, v] -= found_e
        X[I_, v] -= found_i
        X[SEV_, v] -= found_sev
        X[R_, v] += found_e + found_i + found_sev
        self.kits -= 1
        self._last_screen_tick[v] = self.tick
        self.last_positivity[v] = (found_e + found_i + found_sev) / tested
        self.totals["screens"] += 1
        return found_e, found_i, found_sev

    def _restock(self) -> None:
        p = self.params
        self.fuel = p.fuel_capacity_km
        self.kits = p.kit_capacity
        self.nets = p.net_capacity

    def _project_rest_of_season(self) -> tuple[float, float]:
        """Expected infections and severe cases from now to day 30 with no van (mean-field).

        Charged on termination so that ending the episode early (stranding, or letting
        the outbreak escape) is never cheaper than carrying on.
        """
        p = self.params
        X = self.X.astype(np.float64)
        rates = np.array([p.p_early_to_symptomatic, p.p_symptomatic_to_severe, p.p_symptomatic_recover, p.p_severe_recover])
        total_inf = total_sev = 0.0
        for t in range(self.tick, p.max_ticks):
            new_e = X[S_] * -np.expm1(-self._hazard(t, X, self._net_until))
            e_to_i, i_to_sev, i_rec, sev_rec = rates[:, None] * X[[E_, I_, I_, SEV_]]
            X += _FLOW_MATRIX @ np.stack((new_e, e_to_i, i_to_sev, i_rec, sev_rec))
            total_inf += float(new_e.sum())
            total_sev += float(i_to_sev.sum())
        return total_inf, total_sev

    def _observe(self) -> np.ndarray:
        p, t = self.params, self.tick
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[self.location] = 1.0
        obs[OBS_FUEL] = self.fuel / p.fuel_capacity_km
        obs[OBS_KITS] = self.kits / p.kit_capacity
        obs[OBS_NETS] = self.nets / p.net_capacity
        obs[OBS_DAY] = t / p.max_ticks
        obs[OBS_RAIN] = self._rain[t + self._lag_pad]
        obs[OBS_FORECAST] = self._forecast[t]
        obs[OBS_REPORTED] = self.reported / (p.report_scale * POPULATION)
        obs[OBS_DAYS_SINCE_SCREEN] = (t - self._last_screen_tick) / (p.ticks_per_day * DAYS_SINCE_SCREEN_SCALE)
        obs[OBS_POSITIVITY] = self.last_positivity / p.positivity_scale
        obs[OBS_TRAVEL_TIME] = self._travel_table[t, self.location, 1:] / p.max_travel_ticks
        obs[OBS_NET_DAYS_LEFT] = (self._net_until - t) / p.net_ticks
        return np.clip(obs, 0.0, 1.0)

    def _info(self, step_info: dict[str, Any] | None) -> dict[str, Any]:
        info: dict[str, Any] = step_info if step_info is not None else {}
        totals = self.totals
        info.update(
            tick=self.tick,
            day=self.tick / self.params.ticks_per_day,
            location=self.location,
            fuel_km=self.fuel,
            kits=self.kits,
            nets=self.nets,
            district_prevalence=self.district_prevalence(),
            compartments=self.X.copy(),  # rows: COMPARTMENTS, columns: villages
            transmission=self._last_hazard.copy(),  # per-village hazard over the last tick
            hotspots=self.hotspots.copy(),
            episode_infections=totals["infections"] + totals["settled_infections"],
            totals=totals.copy(),
        )
        return info
