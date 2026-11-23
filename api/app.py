"""FastAPI service: the 3D viewer, a live simulation stream, and a recommendation endpoint.

    uv run main.py serve [--port 8000]   # API + viewer
    uv run play.py                        # same app; opens the browser and prints the step log

Endpoints
- ``GET  /``             the Three.js viewer (static files, three.js vendored)
- ``GET  /scene``        the Bugesera map and constants
- ``GET  /model-info``   which trained models are loaded, with their config and held-out results
- ``POST /recommend``    observation (55 floats) -> recommended action, its name, and the values behind it
- ``WS   /ws/sim``       streams one frame per decision; query: algo, seed, scenario, speed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from environment.custom_env import ACTION_NAMES, ENV_VERSION, LOCATION_NAMES, OBS_DIM, OBS_LOCATION, SCENARIOS
from environment.rendering import (
    EpisodeRunner,
    reference_runs,
    rationale,
    scene,
    terminal_header,
    terminal_line,
    terminal_summary,
)
from training.common import ALGOS, ROOT, TorchPolicy, load_policy
from training.seeds import HELD_OUT_SEEDS

VIEWER_DIR = ROOT / "environment" / "viewer"
BEST_FILE = ROOT / "models" / "best.json"  # {"algo": "<overall best algorithm>"}, written with the final models
SHUTDOWN_GRACE_S = 3  # Ctrl+C / SIGTERM never waits longer than this for open viewer streams


def model_family_dir(algo: str) -> Path:
    return ROOT / "models" / ("dqn" if algo == "dqn" else Path("pg") / algo)


def resolve_model_dir(algo: str) -> Path:
    """``best`` -> the overall best algorithm; each algorithm -> its committed best model."""
    if algo == "best":
        algo = json.loads(BEST_FILE.read_text())["algo"] if BEST_FILE.exists() else "ppo"
    best = model_family_dir(algo) / "best"
    if (best / "model.pt").exists():
        return best
    candidates = sorted(model_family_dir(algo).glob("*/model.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1].parent
    raise FileNotFoundError(f"no trained {algo} model under {model_family_dir(algo)}; "
                            f"train one (uv run main.py train --algo {algo}) or pass --model-dir")


@dataclass
class Settings:
    """How the app finds models and how the live stream behaves."""

    model_dirs: dict[str, Path] = field(default_factory=dict)  # algo -> model dir (overrides resolution)
    default_algo: str = "best"
    default_seed: int = HELD_OUT_SEEDS[0]
    default_scenario: str = "default"
    default_speed: float = 1.0
    tick_seconds: float = 0.6  # wall time per half-day tick at speed 1
    verbose: bool = False  # print the step log to the terminal (play.py)
    colour: bool = False
    compare_baselines: bool = True


class RecommendRequest(BaseModel):
    observation: list[float] = Field(..., min_length=OBS_DIM, max_length=OBS_DIM,
                                     description="the 55-value observation, each in [0, 1]")
    algo: str = Field("best", description="best | dqn | reinforce | a2c | ppo")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Afya Sentinel", version=ENV_VERSION,
                  description="Malaria early-detection van dispatch in Bugesera District, Rwanda")
    policies: dict[str, TorchPolicy] = {}

    def model_dir(algo: str) -> Path:
        if algo in settings.model_dirs:
            return Path(settings.model_dirs[algo])
        if algo == "best" and settings.model_dirs and settings.default_algo in settings.model_dirs:
            return Path(settings.model_dirs[settings.default_algo])
        return resolve_model_dir(algo)

    def policy(algo: str) -> TorchPolicy:
        if algo not in policies:
            try:
                policies[algo] = load_policy(model_dir(algo))
            except FileNotFoundError as err:
                raise HTTPException(404, str(err)) from err
        return policies[algo]

    @app.get("/scene")
    def get_scene() -> dict[str, Any]:
        return scene()

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        models = {}
        for algo in ("best", *ALGOS):
            try:
                directory = model_dir(algo)
            except (FileNotFoundError, KeyError):
                continue
            meta = json.loads((directory / "config.json").read_text())
            cfg = meta["config"]
            models[algo] = {"algo": cfg["algo"], "run_id": cfg["run_id"], "seed": cfg["seed"],
                            "frames": meta.get("frames"), "env_version": meta.get("env_version"),
                            "model_dir": str(directory), "config": cfg}
            candidates = [directory / "eval.json", ROOT / "logs" / cfg["algo"] / cfg["run_id"] / "eval.json",
                          Path(cfg.get("logs_root", "")) / cfg["algo"] / cfg["run_id"] / "eval.json"]
            ev_path = next((c for c in candidates if c.is_file()), None)
            if ev_path is not None:
                ev = json.loads(ev_path.read_text())["eval"]
                models[algo]["held_out"] = {k: ev["held_out"][k] for k in
                                            ("return_mean", "return_std", "infections_averted_mean", "policy_mode")}
        return {"env_version": ENV_VERSION, "actions": list(ACTION_NAMES), "models": models}

    @app.post("/recommend")
    def recommend(request: RecommendRequest) -> dict[str, Any]:
        if any(not (0.0 <= v <= 1.0) for v in request.observation):
            raise HTTPException(422, "observation values must be in [0, 1]")
        import numpy as np

        obs = np.asarray(request.observation, dtype=np.float32)
        loaded = policy(request.algo)
        deterministic = TorchPolicy(loaded.actor, loaded.name, sample=False)  # a recommendation is the argmax
        action, thinking = deterministic.act_and_explain(obs)
        location = int(np.argmax(obs[OBS_LOCATION]))
        return {"action": action, "action_name": ACTION_NAMES[action], "location": LOCATION_NAMES[location],
                "kind": thinking["kind"], "values": thinking["values"], "mask": thinking["mask"],
                "rationale": rationale(obs, action, location, thinking), "model": loaded.name}

    app.state.active_streams = 0  # live /ws/sim handlers (a test checks they all end on disconnect)

    @app.websocket("/ws/sim")
    async def simulate(ws: WebSocket) -> None:
        await ws.accept()
        q = ws.query_params
        algo = q.get("algo", settings.default_algo)
        scenario = q.get("scenario", settings.default_scenario)
        if scenario not in SCENARIOS:
            await ws.send_json({"type": "error", "message": f"unknown scenario {scenario!r}"})
            await ws.close()
            return
        seed = int(q.get("seed", settings.default_seed))
        state = {"paused": False, "speed": float(q.get("speed", settings.default_speed)), "next": False}
        try:
            agent = policy(algo)
        except HTTPException as err:
            await ws.send_json({"type": "error", "message": err.detail})
            await ws.close()
            return

        async def receive_controls() -> None:
            while True:
                msg = await ws.receive_json()
                cmd = msg.get("cmd")
                if cmd == "pause":
                    state["paused"] = True
                elif cmd == "resume":
                    state["paused"] = False
                elif cmd == "speed":
                    state["speed"] = float(min(max(msg.get("value", 1.0), 0.1), 50.0))
                elif cmd == "next":
                    state["next"] = True

        controls = asyncio.create_task(receive_controls())

        async def wait_until(condition) -> None:
            """Wait for the viewer (pause, end of season), but stop if it has gone away."""
            while not condition():
                if controls.done():  # the client disconnected (or the server is shutting down)
                    raise WebSocketDisconnect()
                await asyncio.sleep(0.1)

        app.state.active_streams += 1
        try:
            while True:
                runner = EpisodeRunner(agent, seed, scenario, settings.tick_seconds)
                start = runner.reset()
                await ws.send_json({"type": "episode", "seed": seed, "scenario": scenario, "algo": algo,
                                    "model": agent.name, "policy_mode": "sample" if agent.sample else "greedy",
                                    "start": start})
                if settings.verbose:
                    print(terminal_header(agent.name, seed, scenario), flush=True)
                state["next"] = False
                while not runner.done:
                    await wait_until(lambda: not state["paused"] or state["next"])
                    frame = runner.step(state["speed"])
                    if settings.verbose:
                        print(terminal_line(frame, settings.colour), flush=True)
                    await ws.send_json(frame)
                    await asyncio.sleep(frame["duration_s"])
                    if state["next"]:
                        break
                summary = runner.summary()
                summary["reference"] = reference_runs(seed, scenario) if settings.compare_baselines else {}
                if settings.verbose:
                    print(terminal_summary(summary, summary["reference"]), flush=True)
                await ws.send_json(summary)
                await wait_until(lambda: state["next"])  # the viewer asks for the next district
                seed += 1
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            controls.cancel()
            app.state.active_streams -= 1

    app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")
    return app


def serve_main(argv: Sequence[str] | None = None) -> int:
    """``main.py serve``: run the API and viewer without opening a browser."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="main.py serve", description="Serve the viewer and the /recommend API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--algo", default="best", choices=("best", *ALGOS))
    parser.add_argument("--model-dir", default=None, help="serve this model directory as --algo")
    args = parser.parse_args(argv)
    settings = Settings(default_algo=args.algo)
    if args.model_dir:
        settings.model_dirs[args.algo] = Path(args.model_dir)
    print(f"Afya Sentinel API on http://{args.host}:{args.port} (docs at /docs)", flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning",
                timeout_graceful_shutdown=SHUTDOWN_GRACE_S)
    return 0


if __name__ == "__main__":
    sys.exit(serve_main())
