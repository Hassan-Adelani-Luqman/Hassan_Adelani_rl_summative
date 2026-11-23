"""Run training shards on Kaggle CPU kernels.

    uv run python -m cloud.kaggle.launch package          # git archive of HEAD -> build/dataset
    uv run python -m cloud.kaggle.launch dataset          # create/version the private source dataset
    uv run python -m cloud.kaggle.launch push --shard ppo [--frames 300000] [--workers 4]
    uv run python -m cloud.kaggle.launch up --shard test  # package + dataset + push in one go
    uv run python -m cloud.kaggle.launch status [--shard ppo ...]
    uv run python -m cloud.kaggle.launch pull --shard ppo # download outputs, merge into logs/ and models/

The Kaggle CLI runs through ``uvx`` (with ``six``, which kaggle 2.x forgets to declare), so it
is never a dependency of this project. Credentials stay in ~/.kaggle; the username is read
from ``kaggle config view``. Kernels are private, CPU-only and need internet for ``uv sync``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from training.sweeps import SHARDS

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
PULLED = HERE / "pulled"
DATASET_SLUG = "afya-sentinel-source"
KERNEL_PREFIX = "afya-sentinel"
KAGGLE = ["uvx", "--with", "six", "kaggle"]
ARCHIVE_EXCLUDES = ("logs", "models", "assets", "report")  # outputs, not needed to train


def kaggle(*args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*KAGGLE, *args], check=check, text=True, capture_output=capture)


def username() -> str:
    out = kaggle("config", "view", capture=True).stdout
    match = re.search(r"username:\s*(\S+)", out)
    if not match or match.group(1) == "None":
        raise SystemExit("no Kaggle username configured (~/.kaggle)")
    return match.group(1)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


# ------------------------------------------------------------------------------ dataset


def package() -> Path:
    """Archive the committed source (HEAD) for the dataset; warns about uncommitted changes."""
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("warning: uncommitted changes are NOT in the package (it is built from HEAD):\n" + dirty)
    out_dir = BUILD / "dataset"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    excludes = [f":(exclude){path}" for path in ARCHIVE_EXCLUDES]
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(out_dir / "afya_source.tar.gz"), "HEAD", "--", ".",
                    *excludes], cwd=ROOT, check=True)
    (out_dir / "SOURCE_COMMIT").write_text(commit + "\n")
    size_kb = (out_dir / "afya_source.tar.gz").stat().st_size // 1024
    print(f"packaged {commit[:10]} -> {out_dir} ({size_kb} KB)")
    return out_dir


def dataset(message: str | None = None) -> None:
    """Create the private source dataset, or add a new version if it already exists."""
    user = username()
    out_dir = BUILD / "dataset"
    if not (out_dir / "afya_source.tar.gz").exists():
        out_dir = package()
    commit = (out_dir / "SOURCE_COMMIT").read_text().strip()
    meta = {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (out_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", capture=True, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(out_dir), "-m", message or f"source {commit[:10]}", "-q")
    else:
        kaggle("datasets", "create", "-p", str(out_dir), "-q")  # private by default
    wait_for_dataset(user)


def wait_for_dataset(user: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", capture=True, check=False).stdout.strip()
        print(f"dataset status: {status}", flush=True)
        if status.lower() == "ready":
            return
        time.sleep(10)
    raise SystemExit("dataset did not become ready in time")


# ------------------------------------------------------------------------------ kernels


def kernel_ref(user: str, shard: str) -> str:
    return f"{user}/{KERNEL_PREFIX}-{shard}"


def render_kernel(shard: str, frames: int | None, workers: int) -> str:
    """kernel.py with this shard's CONFIG filled in (Kaggle uploads only the code file)."""
    source = (HERE / "kernel.py").read_text()
    config = repr({"shard": shard, "frames": frames, "workers": workers})
    rendered, count = re.subn(r"^CONFIG = .*$", f"CONFIG = {config}  # set by launch.py", source, flags=re.M)
    if count != 1:
        raise SystemExit("could not find the CONFIG line in kernel.py")
    return rendered


def parse_status(text: str) -> str:
    """'user/kernel has status "KernelWorkerStatus.COMPLETE"' -> 'complete'."""
    match = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)"?', text, flags=re.I)
    return match.group(1).lower() if match else text.strip()


def output_roots(shard: str) -> tuple[Path, Path]:
    """Where a shard's pulled logs/models go; the test shard stays out of the real results."""
    if shard == "test":
        return ROOT / "logs" / "smoke" / "kaggle", ROOT / "models" / "smoke" / "kaggle"
    return ROOT / "logs", ROOT / "models"


def push(shard: str, frames: int | None, workers: int) -> None:
    """Render kernel.py with this shard's CONFIG and push it as a private CPU kernel."""
    user = username()
    folder = BUILD / "kernels" / shard
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "kernel.py").write_text(render_kernel(shard, frames, workers))
    meta = {
        "id": kernel_ref(user, shard),
        "title": f"{KERNEL_PREFIX}-{shard}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle("kernels", "push", "-p", str(folder))
    print(f"pushed {meta['id']} (frames={frames}, workers={workers})")


def status(shards: Sequence[str]) -> dict[str, str]:
    user = username()
    states = {}
    for shard in shards:
        out = kaggle("kernels", "status", kernel_ref(user, shard), capture=True, check=False)
        states[shard] = parse_status(out.stdout or out.stderr)
        print(f"{shard:<10} {states[shard]}")
    return states


def pull(shard: str) -> None:
    """Download a kernel's output and merge its logs/ and models/ into the repo's."""
    user = username()
    dest = PULLED / shard
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    kaggle("kernels", "output", kernel_ref(user, shard), "-p", str(dest), "-o")
    out = next((p for p in [dest / "out", dest] if (p / "logs").exists()), None)
    if out is None:
        raise SystemExit(f"no logs/ in the output of {shard}; see {dest}")
    merged = 0
    for sub, root in zip(("logs", "models"), output_roots(shard), strict=True):
        for path in (out / sub).rglob("*"):
            if path.is_file():
                target = root / path.relative_to(out / sub)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                merged += 1
    summary = out / "kernel_summary.json"
    print(f"merged {merged} files from {shard}" + (f"; kernel summary: {summary.read_text()}" if summary.exists() else ""))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cloud.kaggle.launch", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("package", help="archive HEAD for the source dataset")
    ds = sub.add_parser("dataset", help="create or version the private source dataset")
    ds.add_argument("-m", "--message", default=None)
    for name in ("push", "up"):
        p = sub.add_parser(name, help="push a shard kernel" + (" (after package + dataset)" if name == "up" else ""))
        p.add_argument("--shard", choices=SHARDS, required=True)
        p.add_argument("--frames", type=int, default=None)
        p.add_argument("--workers", type=int, default=4)
    st = sub.add_parser("status", help="kernel status per shard")
    st.add_argument("--shard", nargs="+", default=list(SHARDS))
    pl = sub.add_parser("pull", help="download a shard's output and merge it")
    pl.add_argument("--shard", choices=SHARDS, required=True)
    args = parser.parse_args(argv)

    if args.command == "package":
        package()
    elif args.command == "dataset":
        dataset(args.message)
    elif args.command == "push":
        push(args.shard, args.frames, args.workers)
    elif args.command == "up":
        package()
        dataset()
        push(args.shard, args.frames, args.workers)
    elif args.command == "status":
        status(args.shard)
    elif args.command == "pull":
        pull(args.shard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
