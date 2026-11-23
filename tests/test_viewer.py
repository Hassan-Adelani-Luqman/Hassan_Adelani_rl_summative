"""Viewer bridge, API and play.py tests (no browser): frames, endpoints, live stream, CLI."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from analysis.baselines import Greedy
from api.app import Settings, create_app
from environment.custom_env import AfyaSentinelEnv
from environment.rendering import (
    FRAME_KEYS,
    VILLAGE_KEYS,
    EpisodeRunner,
    isfinite_payload,
    scene,
    terminal_line,
)
from training.cli import train
from training.common import default_config, load_policy, run_dirs
from training.seeds import HELD_OUT_SEEDS

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "environment" / "viewer"


@pytest.fixture(scope="module")
def ppo_model(tmp_path_factory) -> Path:
    """A tiny trained PPO model (micro-run) for the viewer and API."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("viewer_model")
    cfg = default_config("ppo", run_id="viewer", total_frames=1024, num_envs=4, rollout_len=32, minibatch_size=64,
                         ppo_epochs=1, eval_limit=1, logs_root=str(root / "logs"), models_root=str(root / "models"))
    train(cfg)
    return run_dirs(cfg)[1]


def _play_episode(policy, seed=HELD_OUT_SEEDS[0]):
    runner = EpisodeRunner(policy, seed)
    frames = [runner.reset()]
    while not runner.done:
        frames.append(runner.step())
    return runner, frames


def test_scene_describes_the_bugesera_map():
    s = scene()
    assert s["district"] == "Bugesera District, Rwanda"
    assert [loc["name"] for loc in s["locations"]][0] == "Nyamata" and s["locations"][0]["hub"]
    assert len(s["locations"]) == 9 and len(s["roads"]) == 13 and len(s["actions"]) == 12
    assert {loc["name"] for loc in s["locations"] if loc["wetland"]} == {"Gashora", "Rweru", "Mareba", "Ntarama"}
    json.dumps(s)


def test_every_frame_of_an_episode_is_complete_json(ppo_model):
    with torch.random.fork_rng():
        runner, frames = _play_episode(load_policy(ppo_model))
    assert frames[0]["type"] == "start" and all(f["type"] == "frame" for f in frames[1:])
    for frame in frames:
        assert set(FRAME_KEYS) <= set(frame), set(FRAME_KEYS) - set(frame)
        assert len(frame["villages"]) == 8 and all(set(VILLAGE_KEYS) <= set(v) for v in frame["villages"])
        assert isfinite_payload(frame)
        json.dumps(frame)
    moves = [f for f in frames[1:] if len(f["route"]) > 1]
    assert moves, "the van never moved"
    assert len({f["location"] for f in frames}) > 1  # the van's position changes across frames
    for f in frames[1:]:
        thinking = f["thinking"]
        assert thinking["kind"] == "probability" and thinking["chosen"] == f["action"]
        assert thinking["values"][f["action"]] is not None  # the chosen action is always a valid one
        assert sum(v for v in thinking["values"] if v is not None) == pytest.approx(1.0, abs=1e-3)
        assert thinking["rationale"]
    summary = runner.summary()
    assert summary["type"] == "end" and summary["decisions"] == len(frames) - 1


def test_viewer_episode_matches_the_evaluated_episode(ppo_model):
    """The viewer plays exactly the episode that evaluation scores for that seed."""
    from analysis.baselines import run_episode

    policy = load_policy(ppo_model)
    with torch.random.fork_rng():
        runner, _ = _play_episode(policy)
        evaluated = run_episode(policy, HELD_OUT_SEEDS[0], "default", AfyaSentinelEnv())
    assert runner.summary()["return"] == pytest.approx(evaluated["return"], abs=0.01)


def test_baseline_policies_play_in_the_viewer_too():
    runner, frames = _play_episode(Greedy())
    assert frames[-1]["thinking"]["values"] is None and frames[-1]["thinking"]["rationale"]
    line = terminal_line(frames[1])
    assert re.match(r"\s*\d+\.\d \| ", line) and "\x1b[" not in line  # no colour codes unless asked


@pytest.fixture()
def client(ppo_model):
    app = create_app(Settings(model_dirs={"ppo": ppo_model}, default_algo="ppo", tick_seconds=0.0,
                              compare_baselines=False))
    with TestClient(app) as c:
        yield c


def test_api_serves_scene_viewer_and_model_info(client):
    assert client.get("/scene").json()["locations"][0]["name"] == "Nyamata"
    page = client.get("/")
    assert page.status_code == 200 and "importmap" in page.text and "main.js" in page.text
    assert client.get("/vendor/three.module.js").status_code == 200
    info = client.get("/model-info").json()
    assert info["models"]["ppo"]["algo"] == "ppo" and info["env_version"] == "1.0"


def test_recommend_returns_an_action_with_its_values(client):
    obs, _ = AfyaSentinelEnv().reset(seed=HELD_OUT_SEEDS[0])
    body = client.post("/recommend", json={"observation": obs.tolist(), "algo": "ppo"}).json()
    assert 0 <= body["action"] < 12 and body["action_name"]
    assert body["kind"] == "probability" and len(body["values"]) == 12 and len(body["mask"]) == 12
    assert body["mask"][body["action"]] and body["location"] == "Nyamata" and body["rationale"]
    # at the hub, screening and nets are masked out
    assert body["values"][8] is None and body["values"][9] is None


@pytest.mark.parametrize("observation", [[0.5] * 54, [2.0] * 55])
def test_recommend_rejects_bad_observations(client, observation):
    assert client.post("/recommend", json={"observation": observation}).status_code == 422


def test_websocket_streams_a_whole_episode(client):
    with client.websocket_connect(f"/ws/sim?algo=ppo&seed={HELD_OUT_SEEDS[1]}&speed=50") as ws:
        first = ws.receive_json()
        assert first["type"] == "episode" and first["start"]["type"] == "start" and first["seed"] == HELD_OUT_SEEDS[1]
        frames, msg = 0, ws.receive_json()
        while msg["type"] == "frame":
            frames += 1
            msg = ws.receive_json()
        assert msg["type"] == "end" and msg["decisions"] == frames > 0
        ws.send_json({"cmd": "next"})
        nxt = ws.receive_json()
        assert nxt["type"] == "episode" and nxt["seed"] == HELD_OUT_SEEDS[1] + 1


async def _stream_until_viewer_leaves(app, query: str, leave_on: str, before_leaving: list[dict]) -> None:
    """Drive /ws/sim as uvicorn would: when the server sends a ``leave_on`` message, the viewer
    sends ``before_leaving`` commands and disconnects. Uvicorn (unlike the TestClient) never
    cancels the handler, so it has to notice the disconnect and return on its own."""
    inbox: asyncio.Queue = asyncio.Queue()
    inbox.put_nowait({"type": "websocket.connect"})
    left = False

    async def send(message):
        nonlocal left
        if message["type"] == "websocket.send" and not left and json.loads(message["text"])["type"] == leave_on:
            for cmd in before_leaving:
                inbox.put_nowait({"type": "websocket.receive", "text": json.dumps(cmd)})
            inbox.put_nowait({"type": "websocket.disconnect", "code": 1001})
            left = True

    scope = {"type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws", "path": "/ws/sim", "raw_path": b"/ws/sim",
             "query_string": query.encode(), "root_path": "", "headers": [], "subprotocols": [],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000)}
    await asyncio.wait_for(app(scope, inbox.get, send), timeout=20)


@pytest.mark.parametrize(("leave_on", "before_leaving"), [("frame", [{"cmd": "pause"}]), ("end", [])],
                         ids=["while_paused", "on_the_end_screen"])
def test_stream_handler_ends_when_the_viewer_leaves(ppo_model, leave_on, before_leaving):
    """A viewer that leaves while paused or on the end screen must not leave its handler waiting
    forever: that blocked server shutdown (Ctrl+C) indefinitely."""
    app = create_app(Settings(model_dirs={"ppo": ppo_model}, default_algo="ppo", tick_seconds=0.0,
                              compare_baselines=False))
    asyncio.run(_stream_until_viewer_leaves(app, "algo=ppo&speed=50", leave_on, before_leaving))
    assert app.state.active_streams == 0


def test_websocket_reports_a_missing_model(tmp_path):
    app = create_app(Settings(model_dirs={"dqn": tmp_path / "no_such_model"}, default_algo="dqn"))
    with TestClient(app) as c, c.websocket_connect("/ws/sim?algo=dqn") as ws:
        message = ws.receive_json()
    assert message["type"] == "error"


def test_play_runs_in_the_terminal(ppo_model):
    result = subprocess.run([sys.executable, "play.py", "--no-browser", "--episodes", "1", "--compare-greedy",
                             "--model-dir", str(ppo_model)], capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "Bugesera District" in result.stdout and "episode over" in result.stdout
    assert "same district, greedy" in result.stdout


def test_viewer_uses_only_local_assets():
    for path in [VIEWER / "index.html", VIEWER / "main.js", VIEWER / "style.css"]:
        # the inline SVG favicon's xmlns (www.w3.org) is a namespace name, never fetched
        assert not re.search(r"https?://(?!www\.w3\.org/)", path.read_text()), f"external URL in {path.name}"
    assert (VIEWER / "vendor" / "three.module.js").exists() and (VIEWER / "vendor" / "LICENSE-three.txt").exists()
