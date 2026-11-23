"""Afya Sentinel command-line entry point.

    uv run main.py                 # same as `uv run play.py`: watch the best trained agent
    uv run main.py play [flags]    # play.py flags, e.g. --algo ppo --seed 7
    uv run main.py train|sweep|evaluate|figures|serve ...

Play flags can also be given directly (`uv run main.py --seed 7`); they are passed to play.py.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

COMMANDS: dict[str, str] = {
    "train": "train one algorithm with one config",
    "sweep": "run a hyperparameter sweep shard",
    "evaluate": "evaluate saved models on the held-out seeds",
    "figures": "regenerate report figures and tables from logs/",
    "serve": "serve the viewer and the /recommend API",
}


# commands handled by another module's own argument parser: command -> (module, function)
DELEGATES: dict[str, tuple[str, str]] = {
    "train": ("training.cli", "train_main"),
    "sweep": ("training.run_sweep", "main"),
    "evaluate": ("training.cli", "evaluate_main"),
    "serve": ("api.app", "serve_main"),
}


def _run_play(argv: Sequence[str]) -> int:
    import play  # imported lazily so `main.py --help` stays fast

    return play.main(argv)


def _not_implemented(command: str) -> Callable[[argparse.Namespace], int]:
    def run(_: argparse.Namespace) -> int:
        raise SystemExit(f"main.py {command}: not implemented yet.")

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Afya Sentinel: RL dispatch of a mobile malaria early-detection van. "
        "With no command, runs the demo (same as play.py).",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.add_parser("play", help="watch a trained agent (default; takes play.py flags)", add_help=False)
    for command, help_text in COMMANDS.items():
        cmd = sub.add_parser(command, help=help_text, description=help_text)
        cmd.set_defaults(func=_not_implemented(command))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return _run_play([])
    if args_list[0] == "play":
        return _run_play(args_list[1:])
    if args_list[0].startswith("-") and args_list[0] not in ("-h", "--help"):
        return _run_play(args_list)
    if args_list[0] in DELEGATES:
        import importlib

        module_name, function = DELEGATES[args_list[0]]
        return getattr(importlib.import_module(module_name), function)(args_list[1:])
    args = build_parser().parse_args(args_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
