"""Deep Q-Network training with TorchRL.

    uv run python -m training.dqn_training --frames 20000 --run-id smoke

Components: ``QValueActor`` over an MLP, ``EGreedyModule`` exploration annealed over a
share of the frame budget, ``DQNLoss`` (optionally Double DQN) with a delayed target
network kept in sync by ``HardUpdate`` or ``SoftUpdate``, and a ``ReplayBuffer`` on
``LazyTensorStorage`` filled by a ``Collector`` over the shared vector environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment.custom_env import OBS_DIM, N_ACTIONS
from training.common import (
    RunConfig,
    RunLogger,
    Timer,
    action_spec,
    default_config,
    evaluate_policy,
    make_env,
    mlp,
    run_dirs,
    save_run,
    TorchPolicy,
)


def build_qvalue_actor(cfg: RunConfig) -> QValueActor:
    return QValueActor(mlp(OBS_DIM, N_ACTIONS, cfg), in_keys=["observation"], spec=action_spec())


def train(cfg: RunConfig) -> dict[str, Any]:
    """Train one DQN run, evaluate it on the held-out seeds and save everything. Returns eval.json content."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    timer = Timer()
    env = make_env(cfg.seed, cfg.scenario, cfg.num_envs, cfg.reward_scale)
    actor = build_qvalue_actor(cfg)
    explore = EGreedyModule(
        env.action_spec,
        eps_init=cfg.eps_init,
        eps_end=cfg.eps_end,
        annealing_num_steps=max(1, int(cfg.eps_fraction * cfg.total_frames)),
    )
    collector = Collector(
        env,
        TensorDictSequential(actor, explore),
        frames_per_batch=cfg.num_envs * cfg.steps_per_collect,
        total_frames=cfg.total_frames,
        init_random_frames=cfg.init_random_frames,
    )
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)
    loss_module = DQNLoss(actor, loss_function="smooth_l1", delay_value=True, double_dqn=cfg.double_dqn,
                          action_space=env.action_spec)
    loss_module.make_value_estimator(gamma=cfg.gamma)
    if cfg.target_update == "hard":
        target_updater = HardUpdate(loss_module, value_network_update_interval=cfg.target_interval)
    else:
        target_updater = SoftUpdate(loss_module, tau=cfg.tau)
    optimiser = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    log_dir, _ = run_dirs(cfg)
    logger = RunLogger(log_dir)
    frames = updates = 0
    for batch in collector:
        frames += batch.numel()
        logger.episodes_from_batch(batch, frames)
        buffer.extend(batch.reshape(-1))
        explore.step(batch.numel())
        if frames < cfg.init_random_frames:
            continue
        losses, q_means, grad_norms = [], [], []
        for _ in range(cfg.updates_per_collect):
            sample = buffer.sample()
            loss = loss_module(sample)["loss"]
            optimiser.zero_grad()
            loss.backward()
            grad_norms.append(torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm))
            optimiser.step()
            target_updater.step()
            updates += 1
            losses.append(loss.detach())
            with torch.no_grad():
                q_means.append(actor(sample.select("observation"))["action_value"].max(-1).values.mean())
        logger.update(
            frames=frames,
            updates=updates,
            loss=torch.stack(losses).mean(),
            q_mean=torch.stack(q_means).mean(),
            grad_norm=torch.stack(grad_norms).mean(),
            epsilon=explore.eps,
            lr=cfg.lr,
        )
    collector.shutdown()
    env.close()
    logger.close()

    train_seconds = timer.elapsed()
    evaluation = evaluate_policy(TorchPolicy(actor.eval(), f"dqn:{cfg.run_id}"), cfg.eval_sets, cfg.eval_limit)
    stats = {"frames": frames, "updates": updates, "episodes": logger.episodes, "train_seconds": round(train_seconds, 1),
             "fps": round(frames / train_seconds, 1), "total_seconds": round(timer.elapsed(), 1)}
    save_run(cfg, {"actor": actor}, stats, evaluation)
    return {"stats": stats, "eval": evaluation, "actor": actor}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DQN on Afya Sentinel.")
    parser.add_argument("--frames", type=int, default=None, help="total frames (default: config)")
    parser.add_argument("--run-id", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    overrides = {"run_id": args.run_id, "seed": args.seed}
    if args.frames:
        overrides["total_frames"] = args.frames
    result = train(default_config("dqn", **overrides))
    held_out = result["eval"]["held_out"]
    print(f"dqn:{args.run_id} {result['stats']} | held-out return {held_out['return_mean']:.1f} ± {held_out['return_std']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
