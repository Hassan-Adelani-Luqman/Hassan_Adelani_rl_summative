"""Smoke tests: pinned stack, CPU-only torch, TorchRL <-> gymnasium bridge, CLI entry points."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _exact_pins() -> dict[str, str]:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    return dict(dep.split("==", 1) for dep in deps if "==" in dep)


def test_pinned_versions_are_installed():
    import tensordict
    import torch
    import torchrl

    pins = _exact_pins()
    assert set(pins) == {"torch", "torchrl", "tensordict"}
    assert torch.__version__.split("+")[0] == pins["torch"]
    assert torchrl.__version__ == pins["torchrl"]
    assert tensordict.__version__ == pins["tensordict"]


def test_torch_is_cpu_only():
    import torch

    assert torch.version.cuda is None
    assert not torch.cuda.is_available()


def test_torchrl_compiled_extension_loads():
    # Built against one specific torch release; a mismatch would fail here.
    importlib.import_module("torchrl._torchrl")


def test_torchrl_wraps_a_gymnasium_env():
    import gymnasium as gym
    from torchrl.envs import Compose, GymWrapper, RewardSum, StepCounter, TransformedEnv
    from torchrl.envs.utils import check_env_specs

    env = TransformedEnv(GymWrapper(gym.make("CartPole-v1")), Compose(StepCounter(), RewardSum()))
    env.set_seed(0)
    check_env_specs(env)
    rollout = env.rollout(20)
    assert rollout.shape[0] >= 1
    assert "episode_reward" in rollout["next"].keys()
    assert "step_count" in rollout["next"].keys()


@pytest.mark.parametrize("package", ["environment", "training", "analysis", "api", "cloud.kaggle"])
def test_project_packages_import(package):
    importlib.import_module(package)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=ROOT, timeout=120
    )


@pytest.mark.parametrize("script", ["main.py", "play.py"])
def test_cli_help_exits_cleanly(script):
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_main_lists_every_subcommand():
    out = _run("main.py", "--help").stdout
    for command in ("play", "train", "sweep", "evaluate", "figures", "serve"):
        assert command in out


@pytest.mark.parametrize(
    "argv",
    [
        ["main.py", "play", "--help"],
        ["main.py", "--seed", "3", "--help"],
    ],
)
def test_main_routes_to_play(argv):
    # play's own parser answers, which proves main.py handed the arguments to play.py
    result = _run(*argv)
    assert result.returncode == 0, result.stderr
    assert "usage: play.py" in result.stdout


def test_main_with_no_arguments_starts_play():
    import main

    called = {}
    original = main._run_play
    main._run_play = lambda argv: called.setdefault("argv", list(argv)) or 0
    try:
        assert main.main([]) == 0
    finally:
        main._run_play = original
    assert called["argv"] == []
