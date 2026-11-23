"""Entry point that runs inside a Kaggle CPU kernel: install the locked environment, train one shard.

``launch.py push`` fills in ``CONFIG`` before uploading (Kaggle uploads only this file).
The source comes from the private source dataset attached to the kernel. The repo and its
virtualenv live under /tmp, so the only kernel output is /kaggle/working/out, laid out
like the repo (logs/ and models/), which ``launch.py pull`` downloads and merges.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CONFIG = {"shard": "test", "frames": None, "workers": 4}  # replaced by launch.py

INPUT = Path("/kaggle/input")
SOURCE = Path("/tmp/afya_sentinel")
OUT = Path("/kaggle/working/out")
ENV = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "UV_LINK_MODE": "copy", "OMP_NUM_THREADS": "1",
       "RL_WARNINGS": "False"}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=SOURCE, env=ENV, **kwargs)


def unpack_source() -> str:
    """Extract the source tarball (or copy an already-unpacked tree) into SOURCE; returns its commit."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(INPUT.rglob("afya_source.tar.gz"))
    if tarballs:
        with tarfile.open(tarballs[0]) as tar:
            tar.extractall(SOURCE, filter="data")
    else:  # Kaggle sometimes unpacks archives itself
        trees = [p.parent for p in INPUT.rglob("pyproject.toml") if 'name = "afya-sentinel"' in p.read_text()]
        if not trees:
            listing = [str(p) for p in list(INPUT.rglob("*"))[:60]]
            raise SystemExit(f"source not found under {INPUT}; saw {listing}")
        shutil.copytree(trees[0], SOURCE, dirs_exist_ok=True)
    commits = sorted(INPUT.rglob("SOURCE_COMMIT"))
    return commits[0].read_text().strip() if commits else "unknown"


def main() -> int:
    started = time.time()
    commit = unpack_source()
    print(f"shard {CONFIG['shard']} | source commit {commit} | cpus {os.cpu_count()}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    uv = [sys.executable, "-m", "uv"]
    run([*uv, "sync", "--frozen"], check=True)
    run([*uv, "run", "python", "-c",
         "import torch, torchrl, tensordict, gymnasium, platform; "
         "print('versions: torch', torch.__version__, '| torchrl', torchrl.__version__, '| tensordict', "
         "tensordict.__version__, '| gymnasium', gymnasium.__version__, '| python', platform.python_version())"],
        check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [*uv, "run", "python", "-m", "training.run_sweep", "--shard", CONFIG["shard"],
           "--workers", str(CONFIG["workers"]), "--logs-root", str(OUT / "logs"), "--models-root", str(OUT / "models")]
    if CONFIG["frames"]:
        cmd += ["--frames", str(CONFIG["frames"])]
    result = run(cmd)
    summary = {"config": CONFIG, "commit": commit, "exit_code": result.returncode, "cpus": os.cpu_count(),
               "minutes": round((time.time() - started) / 60, 1)}
    (OUT / "kernel_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", json.dumps(summary), flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
