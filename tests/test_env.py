"""Phase 2 environment tests: API compliance, observation bounds, reward invariants,
termination, semi-MDP timing, determinism, the delayed effects (nets, rain lag), and
the baseline ordering."""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from analysis.baselines import DoNothing, Greedy, RandomPolicy, evaluate
from environment.custom_env import (
    A_HUB,
    A_PREVENT,
    A_SCREEN,
    A_WAIT,
    ACTION_NAMES,
    E_,
    HUB,
    I_,
    N_ACTIONS,
    N_LOCATIONS,
    OBS_DIM,
    OBS_LOCATION,
    OBS_NET_DAYS_LEFT,
    POPULATION,
    R_,
    ROAD_KM,
    ROUTE_FLOOD_FRAC,
    S_,
    SCENARIOS,
    SEV_,
    VILLAGE_NAMES,
    AfyaSentinelEnv,
    EnvParams,
    RewardConfig,
)
from training.seeds import EVAL_SETS, HELD_OUT_SEEDS, TRAIN_EVAL_SEEDS, TRAIN_SEED_POOL, seed_hash

FARTHEST_VILLAGE_ACTION = int(np.argmax(ROAD_KM[HUB, 1:]))  # travel action id == village index


def _assert_valid_obs(env: AfyaSentinelEnv, obs: np.ndarray) -> None:
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert obs[OBS_LOCATION].sum() == 1.0


# ------------------------------------------------------------------------------ API


def test_passes_the_gymnasium_env_checker():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=".*alternative render modes.*")
        check_env(AfyaSentinelEnv())


def test_spaces_and_action_names():
    env = AfyaSentinelEnv()
    assert env.action_space.n == N_ACTIONS == len(ACTION_NAMES) == 12
    assert env.observation_space.shape == (OBS_DIM,) == (55,)
    assert OBS_NET_DAYS_LEFT.stop == OBS_DIM


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_observations_stay_in_bounds(scenario):
    env = AfyaSentinelEnv(scenario)
    for policy in (RandomPolicy(), Greedy()):
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            policy.reset(seed)
            _assert_valid_obs(env, obs)
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(policy(obs))
                _assert_valid_obs(env, obs)
                assert (info["compartments"].sum(axis=0) == POPULATION).all()  # people are conserved
                assert (info["compartments"] >= 0).all()
                done = terminated or truncated


def test_ansi_render_lists_every_village():
    env = AfyaSentinelEnv(render_mode="ansi")
    env.reset(seed=0)
    env.step(A_WAIT)
    text = env.render()
    assert all(name in text for name in VILLAGE_NAMES)


# ------------------------------------------------------------------------------ reward


def test_no_case_is_ever_profitable():
    rw = RewardConfig()
    # A case pays at most one detection bonus (detection treats it), so each bonus must be
    # covered by the penalties that case has already cost.
    assert rw.early_detected <= rw.new_infection
    assert rw.symptomatic_detected <= rw.new_infection
    assert rw.severe_referred <= rw.new_infection + rw.new_severe
    # Earlier detection always loses less.
    net_early = rw.new_infection - rw.early_detected
    net_symptomatic = rw.new_infection - rw.symptomatic_detected
    net_severe = rw.new_infection + rw.new_severe - rw.severe_referred
    assert 0 <= net_early <= net_symptomatic <= net_severe


def test_screening_treats_the_cases_it_finds():
    env = AfyaSentinelEnv()
    env.reset(seed=4)
    v = int(env.hotspots[0])
    env.step(v)  # drive to a hotspot
    before = env.X.copy()
    kits = env.kits
    _, _, _, _, info = env.step(A_SCREEN)
    found = info["early_detected"] + info["symptomatic_detected"] + info["severe_referred"]
    assert found > 0
    assert env.kits == kits - 1
    assert info["totals"]["screens"] == 1
    # found cases moved to R (on top of one tick of natural dynamics, which also conserves people)
    assert env.X[R_, v] >= before[R_, v] + found
    assert env.X[:, v].sum() == POPULATION[v]


@pytest.mark.parametrize(
    "setup, action",
    [
        (lambda env: None, A_SCREEN),  # screening at the hub
        (lambda env: None, A_PREVENT),  # nets at the hub
        (lambda env: env.step(0), 0),  # driving to where the van already is
        (lambda env: (env.step(0), setattr(env, "kits", 0)), A_SCREEN),  # no kits left
        (lambda env: (env.step(0), setattr(env, "nets", 0)), A_PREVENT),  # no nets left
    ],
)
def test_invalid_actions_cost_a_tick_and_a_penalty(setup, action):
    env = AfyaSentinelEnv()
    env.reset(seed=1)
    setup(env)
    tick = env.tick
    _, reward, _, _, info = env.step(action)
    assert info["invalid"]
    assert info["ticks_elapsed"] == 1 and env.tick == tick + 1
    assert reward == pytest.approx(sum(info["tick_rewards"]))
    rw = RewardConfig()
    assert info["tick_rewards"][0] == pytest.approx(
        -rw.invalid - rw.new_infection * info["new_infections"] - rw.new_severe * info["new_severe"]
    )


# ------------------------------------------------------------------------------ termination


def test_running_out_of_fuel_strands_the_van():
    env = AfyaSentinelEnv()
    env.reset(seed=3)
    env.fuel = 1.0
    _, reward, terminated, truncated, info = env.step(FARTHEST_VILLAGE_ACTION)
    assert terminated and not truncated
    assert info["termination_reason"] == "stranded"
    assert info["terminal_penalty"] == -RewardConfig().stranded
    assert info["settled_penalty"] < 0  # the rest of the season is charged
    assert reward == pytest.approx(sum(info["tick_rewards"]) + info["terminal_penalty"] + info["settled_penalty"])
    assert env.location == HUB and env.fuel == 0.0  # it never arrived
    with pytest.raises(RuntimeError):
        env.step(A_WAIT)


def test_an_escaped_outbreak_terminates():
    env = AfyaSentinelEnv()
    env.reset(seed=3)
    infected = (0.3 * POPULATION).astype(np.int64)
    env.X[:] = 0
    env.X[S_] = POPULATION - infected
    env.X[E_] = infected
    _, _, terminated, truncated, info = env.step(A_WAIT)
    assert terminated and not truncated
    assert info["termination_reason"] == "outbreak"
    assert info["terminal_penalty"] == -RewardConfig().outbreak


def test_day_30_truncates_without_terminating():
    env = AfyaSentinelEnv(params=replace(EnvParams(), importation=0.0))
    env.reset(seed=0)
    env.X[S_] = POPULATION  # no infection at all, so no outbreak can end the episode early
    env.X[E_ : SEV_ + 1] = 0
    for k in range(60):
        _, _, terminated, truncated, info = env.step(A_WAIT)
        assert not terminated
        assert truncated == (k == 59)
    assert info["tick"] == 60 and info["termination_reason"] == "time"


def test_settling_the_season_charges_the_expected_do_nothing_cost():
    """Ending early is charged the projected rest of the season, which must match the
    average of actually doing nothing from the same state (so it is never a shortcut)."""
    env = AfyaSentinelEnv()
    env.reset(seed=11)
    for _ in range(10):
        env.step(A_WAIT)
    projected, _ = env._project_rest_of_season()
    simulated = []
    for k in range(60):
        twin = copy.deepcopy(env)
        twin.np_random = np.random.default_rng(k)
        infections, done = 0, False
        while not done:
            _, _, terminated, truncated, info = twin.step(A_WAIT)
            infections += info["new_infections"] + info["settled_infections"]
            done = terminated or truncated
        simulated.append(infections)
    assert projected == pytest.approx(np.mean(simulated), rel=0.10)


# ------------------------------------------------------------------------------ semi-MDP timing


def test_travel_takes_one_to_four_ticks_and_longer_when_flooded():
    env = AfyaSentinelEnv()
    env.reset(seed=0)
    slower_somewhere = False
    for src in range(N_LOCATIONS):
        for dst in range(N_LOCATIONS):
            if src == dst:
                continue
            dry, wet = env.travel_ticks(src, dst, 0.0), env.travel_ticks(src, dst, 1.0)
            assert 1 <= dry <= wet <= 4
            if ROUTE_FLOOD_FRAC[src, dst] == 0:
                assert dry == wet
            slower_somewhere |= wet > dry
    assert slower_somewhere
    # the per-tick lookup table agrees with the formula at every tick
    for t in range(env.params.max_ticks + 1):
        for src in range(N_LOCATIONS):
            for dst in range(N_LOCATIONS):
                assert env._travel_table[t, src, dst] == env.travel_ticks(src, dst, env.rain_at(t))


@pytest.mark.parametrize("seed", range(5))
def test_a_decision_spans_its_ticks_and_sums_their_rewards(seed):
    env = AfyaSentinelEnv()
    env.reset(seed=seed)
    dest = FARTHEST_VILLAGE_ACTION + 1
    expected = env.travel_ticks(HUB, dest, env.rain_at(0))
    obs, reward, _, _, info = env.step(FARTHEST_VILLAGE_ACTION)
    assert info["ticks_elapsed"] == expected == env.tick
    assert len(info["tick_rewards"]) == expected
    assert reward == pytest.approx(sum(info["tick_rewards"]))
    assert env.location == dest and obs[dest] == 1.0
    assert info["km"] == pytest.approx(ROAD_KM[HUB, dest])


def test_returning_to_the_hub_restocks():
    env = AfyaSentinelEnv()
    env.reset(seed=2)
    env.step(0)
    env.step(A_SCREEN)
    env.step(A_PREVENT)
    env.step(A_HUB)
    p = env.params
    assert env.location == HUB
    assert (env.fuel, env.kits, env.nets) == (p.fuel_capacity_km, p.kit_capacity, p.net_capacity)


# ------------------------------------------------------------------------------ determinism


def _rollout(seed: int, policy) -> list[tuple]:
    env = AfyaSentinelEnv()
    obs, _ = env.reset(seed=seed)
    policy.reset(seed)
    trace, done = [(obs,)], False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        trace.append((obs, reward, terminated, truncated, info["compartments"], env.reported.copy()))
        done = terminated or truncated
    return trace


@pytest.mark.parametrize("policy_cls", [Greedy, RandomPolicy])
def test_same_seed_and_actions_give_identical_trajectories(policy_cls):
    a, b = _rollout(123, policy_cls()), _rollout(123, policy_cls())
    assert len(a) == len(b)
    for step_a, step_b in zip(a, b, strict=True):
        for x, y in zip(step_a, step_b, strict=True):
            np.testing.assert_array_equal(x, y)


def test_different_seeds_give_different_districts():
    env = AfyaSentinelEnv()
    first = env.reset(seed=0)[1]["compartments"]
    second = env.reset(seed=1)[1]["compartments"]
    assert not np.array_equal(first, second)


# ------------------------------------------------------------------------------ delayed effects


def test_nets_cut_transmission_for_ten_days():
    p = EnvParams()
    env = AfyaSentinelEnv()
    env.reset(seed=0)
    v = int(env.hotspots[0])
    env.step(v)
    obs, *_ = env.step(A_PREVENT)
    assert obs[OBS_NET_DAYS_LEFT][v] == pytest.approx((p.net_ticks - 1) / p.net_ticks)
    ratio = env.infection_hazard()[v] / env.infection_hazard(with_nets=False)[v]
    assert ratio == pytest.approx(1.0 - p.net_efficacy)
    for _ in range(p.net_ticks - 1):
        env.step(A_WAIT)
    assert env.net_ticks_left[v] == 0  # 20 ticks = 10 days after placement
    assert env.infection_hazard()[v] == pytest.approx(env.infection_hazard(with_nets=False)[v])


def test_nets_reduce_later_infections_in_the_protected_village():
    infections = {True: 0, False: 0}
    for seed in range(30):
        for use_nets in (True, False):
            env = AfyaSentinelEnv()
            env.reset(seed=seed)
            v = int(env.hotspots[0])
            env.step(v)
            susceptible_before = env.X[S_, v]
            env.step(A_PREVENT if use_nets else A_WAIT)  # both take one tick and the same draws
            for _ in range(19):
                env.step(A_WAIT)
            infections[use_nets] += susceptible_before - env.X[S_, v]
    assert infections[True] < 0.8 * infections[False]


def test_transmission_follows_rain_about_five_days_later():
    p = EnvParams()
    env = AfyaSentinelEnv()
    env.reset(seed=0)
    spike = 20
    env._rain[:] = 0.0  # replace the weather with one rainy half-day at tick 20
    env._rain[spike + env._lag_pad] = 1.0
    env._precompute_weather()
    mult = np.array([env.rain_multiplier(t) for t in range(p.max_ticks + 1)])
    raised = np.flatnonzero(mult > p.rain_mult_floor + 1e-12)
    lo, hi = p.rain_lag_ticks
    assert raised.min() == spike + lo and raised.max() == spike + hi  # 4–6 days later
    assert mult[spike] == pytest.approx(p.rain_mult_floor)  # nothing on the day itself


def test_the_forecast_predicts_the_next_three_days_of_rain():
    env = AfyaSentinelEnv()
    env.reset(seed=5)
    env._forecast_noise[:] = 0.0
    env._precompute_weather()
    for t in range(env.params.max_ticks + 1):
        future = [env.rain_at(t + k) for k in range(1, env.params.forecast_ticks + 1)]
        assert env.forecast_at(t) == pytest.approx(np.mean(future))


def test_chw_reports_refresh_once_a_day():
    env = AfyaSentinelEnv()
    env.reset(seed=6)
    day0 = env.reported
    env.step(A_WAIT)  # tick 1: half a day later, still yesterday's report
    assert env.reported is day0
    env.step(A_WAIT)  # tick 2: a new daily report is drawn
    day1 = env.reported
    assert day1 is not day0
    env.step(A_WAIT)  # tick 3: unchanged until the next day
    assert env.reported is day1


# ------------------------------------------------------------------------------ seeds and baselines


def test_seed_sets_are_fixed_and_disjoint():
    assert len(HELD_OUT_SEEDS) == len(set(HELD_OUT_SEEDS)) == 50
    assert not any(seed in TRAIN_SEED_POOL for seed in HELD_OUT_SEEDS)
    assert all(seed in TRAIN_SEED_POOL for seed in TRAIN_EVAL_SEEDS)
    assert seed_hash(HELD_OUT_SEEDS) == seed_hash(list(HELD_OUT_SEEDS)) != seed_hash(TRAIN_EVAL_SEEDS)
    assert set(EVAL_SETS) == {"train", "held_out", "heavy_rain", "shuffled_hotspots"}


def test_greedy_beats_random_and_do_nothing():
    seeds = HELD_OUT_SEEDS[:10]
    results = {cls.name: evaluate(cls(), seeds) for cls in (DoNothing, RandomPolicy, Greedy)}
    greedy = results["greedy"]
    for other in ("do_nothing", "random"):
        assert greedy["return"].mean() > results[other]["return"].mean()
        assert greedy["infections"].mean() < results[other]["infections"].mean()
