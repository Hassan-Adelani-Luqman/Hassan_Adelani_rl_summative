"""Kaggle launcher helpers (no network): kernel rendering, status parsing, output routing."""

from __future__ import annotations

import ast

from cloud.kaggle.launch import ROOT, output_roots, parse_status, render_kernel


def test_rendered_kernel_carries_the_shard_config_and_is_valid_python():
    source = render_kernel("ppo", 300_000, 4)
    ast.parse(source)
    config_lines = [line for line in source.splitlines() if line.startswith("CONFIG = ")]
    assert len(config_lines) == 1
    config = ast.literal_eval(config_lines[0].split("=", 1)[1].split("#")[0].strip())
    assert config == {"shard": "ppo", "frames": 300_000, "workers": 4}
    assert ast.literal_eval(render_kernel("test", None, 4).split("CONFIG = ")[1].split("  #")[0])["frames"] is None


def test_kernel_status_is_parsed_from_the_cli_text():
    assert parse_status('user/afya-sentinel-ppo has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert parse_status('user/afya-sentinel-ppo has status "KernelWorkerStatus.RUNNING"\n') == "running"
    assert parse_status('user/x has status "error"') == "error"


def test_test_shard_output_stays_out_of_the_real_results():
    assert output_roots("ppo") == (ROOT / "logs", ROOT / "models")
    logs, models = output_roots("test")
    assert "smoke" in logs.parts and "smoke" in models.parts
