import json

import numpy as np
import open3d as o3d

from holux_metric.config import load_config
from holux_metric.inspect import inspect_point_cloud
from holux_metric.io import load_point_cloud
from holux_metric.pipeline import run_o3d1


def test_inspect(synthetic_ply):
    info = inspect_point_cloud(load_point_cloud(synthetic_ply))
    assert info["num_points"] == 35300
    assert info["has_colors"] and not info["has_normals"]
    assert info["invalid_points"] == 0


def test_o3d1_end_to_end(synthetic_ply, tmp_path):
    out = tmp_path / "o3d-1"
    qa = run_o3d1(synthetic_ply, load_config(None), out, make_preview=True)

    assert qa["status"] in ("PASS", "WARN")
    assert qa["planes_detected"] >= 2          # piso + pared
    assert qa["output_points"] < qa["input_points"]
    assert qa["colors_preserved"] and qa["finite_coordinates"]

    for name in ["visual_reference_clean.ply", "inspect.json", "cleaning.json", "planes.json", "qa.json"]:
        assert (out / name).is_file(), name
    for name in ["input", "cleaned", "outliers", "normals", "planes"]:
        assert (out / "preview" / f"{name}.png").is_file(), name

    cleaned = o3d.io.read_point_cloud(str(out / "visual_reference_clean.ply"))
    assert cleaned.has_normals() and cleaned.has_colors()
    assert json.loads((out / "qa.json").read_text())["Open3D_version"]


def test_deterministic_counts(synthetic_ply, tmp_path):
    cfg = load_config(None)
    a = run_o3d1(synthetic_ply, cfg, tmp_path / "a", make_preview=False)
    b = run_o3d1(synthetic_ply, cfg, tmp_path / "b", make_preview=False)
    assert a["output_points"] == b["output_points"]


def test_missing_input_fails_cleanly(tmp_path):
    qa = run_o3d1(tmp_path / "nope.ply", load_config(None), tmp_path / "out", make_preview=False)
    assert qa["status"] == "FAIL" and qa["input_readable"] is False
    assert (tmp_path / "out" / "qa.json").is_file()
