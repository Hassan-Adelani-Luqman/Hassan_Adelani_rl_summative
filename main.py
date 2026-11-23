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

# command -> (phase that implements it, help text)
COMMANDS: dict[str, tuple[int, str]] = {
    "train": (3, "train one algorithm with one config"),
    "sweep": (3, "run a hyperparameter sweep shard"),
    "evaluate": (3, "evaluate saved models on the held-out seeds"),
    "figures": (9, "regenerate report figures and tables from logs/"),
    "serve": (7, "serve the viewer and the /recommend API"),
}


def _run_play(argv: Sequence[str]) -> int:
    import play  # imported lazily so `main.py --help` stays fast

    return play.main(argv)


def _not_implemented(command: str, phase: int) -> Callable[[argparse.Namespace], int]:
    def run(_: argparse.Namespace) -> int:
        raise SystemExit(f"main.py {command}: not implemented yet (Phase {phase}).")

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Afya Sentinel: RL dispatch of a mobile malaria early-detection van. "
        "With no command, runs the demo (same as play.py).",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.add_parser("play", help="watch a trained agent (default; takes play.py flags)", add_help=False)
    for command, (phase, help_text) in COMMANDS.items():
        cmd = sub.add_parser(command, help=help_text, description=help_text)
        cmd.set_defaults(func=_not_implemented(command, phase))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return _run_play([])
    if args_list[0] == "play":
        return _run_play(args_list[1:])
    if args_list[0].startswith("-") and args_list[0] not in ("-h", "--help"):
        return _run_play(args_list)
    args = build_parser().parse_args(args_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
