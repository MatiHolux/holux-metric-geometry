"""Pruebas de calibración y métricas (numpy puro: no necesitan Open3D)."""
import numpy as np
import pytest

from holux_metric.calibration import (
    CalibrationError,
    apply_transform,
    calibrate_points,
    estimate_scale,
    parse_references,
    rotation_from_to,
)
from holux_metric.config import DEFAULTS
from holux_metric.measurements import evaluate

CFG = DEFAULTS["calibration"]
K = 3.0  # 1 metro real = 1/3 de unidad visual... (s verdadero = K)


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


class Scene:
    """Escena en metros (Q) y su versión 'visual' P = R Q / K + off (escala y pose desconocidas)."""

    def __init__(self, seed=0, flip_normal=False):
        rng = np.random.default_rng(seed)
        self.R = _random_rotation(rng)
        self.off = rng.normal(size=3) * 5
        floor = np.column_stack([rng.uniform(0, 10, 3000), rng.uniform(0, 6, 3000), np.zeros(3000)])
        wall = np.column_stack([np.zeros(3000), rng.uniform(0, 6, 3000), rng.uniform(0, 3, 3000)])
        self.Q = np.vstack([floor, wall])
        self.P = self.to_visual(self.Q)
        n = self.R @ np.array([0.0, 0.0, 1.0])
        d = -float(n @ self.to_visual(np.zeros((1, 3)))[0])
        if flip_normal:
            n, d = -n, -d
        self.planes = {"planes": [{"id": "plane_001", "normal": n.tolist(), "offset_d": d}]}

    def to_visual(self, Q):
        return (np.atleast_2d(Q) @ self.R.T) / K + self.off

    def ref(self, name, qa, qb, real, role):
        a, b = self.to_visual(qa)[0], self.to_visual(qb)[0]
        return {"name": name, "point_a": a.tolist(), "point_b": b.tolist(), "real_distance_m": real, "role": role}

    def references(self, validation_real=2.2):
        return [
            self.ref("puerta", [1, 1, 0], [2.2, 1, 0], 1.2, "calibrate"),
            self.ref("pared", [0, 0, 0], [6, 0, 0], 6.0, "calibrate"),
            self.ref("altura", [3, 2, 0], [3, 2, 2.2], validation_real, "validate"),
        ]


def test_scale_is_recovered():
    sc = Scene()
    s, diag, warnings = estimate_scale(parse_references({"references": sc.references()})[0], 2.0)
    assert s == pytest.approx(K)
    assert diag["n_calibration_refs"] == 2
    assert warnings == []


@pytest.mark.parametrize("flip", [False, True])
def test_full_frame_recovers_site_coordinates(flip):
    sc = Scene(flip_normal=flip)
    frame = {
        "floor_plane_id": "plane_001",
        "x_axis": {"from": sc.to_visual([0, 0, 0])[0].tolist(), "to": sc.to_visual([1, 0, 0])[0].tolist()},
        "origin_visual": sc.to_visual([0, 0, 0])[0].tolist(),
    }
    doc = {"references": sc.references(), "frame": frame}
    result, (s, R, t), ev = calibrate_points(sc.P, doc, sc.planes, CFG)

    assert s == pytest.approx(K)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    back = apply_transform(sc.P, s, R, t)
    assert np.allclose(back, sc.Q, atol=1e-8)          # recuperamos las coordenadas en metros
    assert ev["summary_validation"]["MAE_m"] < 1e-9
    assert result["frame"]["floor_normal_flipped"] is flip
    assert not any("sobre el plano" in w for w in result["warnings"])   # un piso real no debe dar falsa alarma


def test_scene_ends_above_floor_without_x_axis():
    sc = Scene(seed=3, flip_normal=True)
    doc = {"references": sc.references(), "frame": {"floor_plane_id": "plane_001"}}
    _, (s, R, t), _ = calibrate_points(sc.P, doc, sc.planes, CFG)
    z = apply_transform(sc.P, s, R, t)[:, 2]
    assert z.min() > -1e-8                     # nada queda bajo el piso
    assert 2.9 < z.max() <= 3.0 + 1e-8         # la pared mide 3 m (puntos muestreados al azar)


def test_validation_error_is_reported():
    sc = Scene()
    doc = {"references": sc.references(validation_real=2.25)}
    _, _, ev = calibrate_points(sc.P, doc, None, CFG)
    row = next(r for r in ev["measurements"] if r["name"] == "altura")
    assert row["error_m"] == pytest.approx(-0.05)
    assert ev["summary_validation"]["n"] == 1


def test_warnings_for_weak_setups():
    sc = Scene()
    doc = {"references": [sc.references()[0]]}          # una sola referencia, sin validación ni piso
    result, _, _ = calibrate_points(sc.P, doc, None, CFG)
    text = " ".join(result["warnings"])
    assert "una referencia" in text and "validate" in text and "Z del Site" in text
    assert result["status"] == "WARN"


def test_inconsistent_references_warn():
    sc = Scene()
    refs = sc.references()
    refs[1]["real_distance_m"] = 6.6                     # 10% de diferencia
    _, diag, warnings = estimate_scale(parse_references({"references": refs})[0], 2.0)
    assert diag["ratio_cv_percent"] > 2.0
    assert any("consistentes" in w for w in warnings)


def test_plane_that_splits_the_scene_warns():
    sc = Scene()
    n = np.array([1.0, 0.0, 0.0])
    mid = np.median(sc.P, axis=0)
    planes = {"planes": [{"id": "plane_009", "normal": n.tolist(), "offset_d": -float(n @ mid)}]}
    doc = {"references": sc.references(), "frame": {"floor_plane_id": "plane_009"}}
    # el plano pasa por la mediana: deja ~50% a cada lado
    result, _, _ = calibrate_points(sc.P, doc, planes, CFG)
    assert any("sobre el plano" in w for w in result["warnings"])


def test_input_errors():
    sc = Scene()
    with pytest.raises(CalibrationError):
        calibrate_points(sc.P, {"references": [sc.references()[2]]}, None, CFG)   # sin role: calibrate
    bad = sc.references()
    bad[0]["real_distance_m"] = None
    with pytest.raises(CalibrationError):
        parse_references({"references": bad})
    with pytest.raises(CalibrationError):
        calibrate_points(sc.P, {"references": sc.references(), "frame": {"floor_plane_id": "plane_001"}}, None, CFG)
    with pytest.raises(CalibrationError):
        calibrate_points(sc.P, {"references": sc.references(), "frame": {"floor_plane_id": "nope"}}, sc.planes, CFG)


def test_rotation_from_to():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = rotation_from_to(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
    R = rotation_from_to(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]))
    assert np.allclose(R @ np.array([0, 0, 1.0]), [0, 0, -1.0])


def test_metric_statistics():
    gt = [1.0, 2.0, 3.0, 4.0, 5.0]
    errs = [0.01, -0.02, 0.03, -0.04, 0.05]
    rows = [{"name": f"r{i}", "role": "validate", "ground_truth_m": g, "model_m": g + e}
            for i, (g, e) in enumerate(zip(gt, errs))]
    s = evaluate(rows)["summary_validation"]
    assert s["MAE_m"] == pytest.approx(0.03)
    assert s["RMSE_m"] == pytest.approx(np.sqrt(np.mean(np.square(errs))))
    assert s["max_abs_error_m"] == pytest.approx(0.05)
    assert s["P95_abs_error_m"] == pytest.approx(np.percentile(np.abs(errs), 95))
    assert s["bias_m"] == pytest.approx(0.006)
