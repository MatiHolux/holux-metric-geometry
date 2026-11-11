"""Calibración de punta a punta sobre un .ply (requiere Open3D)."""
import json

import numpy as np
import open3d as o3d

from holux_metric.calibration import run_calibration
from holux_metric.config import load_config

REFS = """
references:
  - {name: largo_a, point_a: [0, 0, 0], point_b: [5, 0, 0], real_distance_m: 10.0, role: calibrate}
  - {name: largo_b, point_a: [0, 5, 0], point_b: [5, 5, 0], real_distance_m: 10.0, role: calibrate}
  - {name: alto,    point_a: [0, 0, 0], point_b: [0, 0, 3], real_distance_m: 6.0,  role: validate}
frame:
  floor_plane_id: plane_001
"""


def test_run_calibration_end_to_end(synthetic_ply, tmp_path):
    refs = tmp_path / "refs.yaml"
    refs.write_text(REFS, encoding="utf-8")
    planes = tmp_path / "planes.json"
    planes.write_text(json.dumps({"planes": [{"id": "plane_001", "normal": [0, 0, 1], "offset_d": 0.0}]}))

    out = tmp_path / "calibration"
    res = run_calibration(synthetic_ply, refs, planes, load_config(None), out)

    cal, meas = res["calibration"], res["measurements"]
    assert abs(cal["scale"] - 2.0) < 1e-9
    assert meas["summary_validation"]["MAE_m"] < 1e-9
    for name in ["site_reference_metric.ply", "site_calibration.json", "measurements.json"]:
        assert (out / name).is_file(), name

    metric = o3d.io.read_point_cloud(str(out / "site_reference_metric.ply"))
    pts = np.asarray(metric.points)
    # el piso (z ~ 0 original) queda en z ~ 0 y la pared de 3 m queda de ~6 m
    assert abs(np.median(pts[:, 2][pts[:, 2] < 0.5])) < 0.05
    assert pts[:, 2].max() > 5.5
