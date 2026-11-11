"""METRIC QUALITY: compara medidas reales (ground truth) vs medidas del modelo.

Estadísticos: MAE, RMSE, error máximo y P95 (sobre el error absoluto).
Los estadísticos "oficiales" se calculan solo con referencias de validación,
es decir, medidas que NO se usaron para calibrar.
"""
from __future__ import annotations

import numpy as np


def _stats(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    err = np.array([r["error_m"] for r in rows], dtype=float)
    ae = np.abs(err)
    gt = np.array([r["ground_truth_m"] for r in rows], dtype=float)
    return {
        "n": int(len(rows)),
        "MAE_m": float(ae.mean()),
        "RMSE_m": float(np.sqrt((err ** 2).mean())),
        "max_abs_error_m": float(ae.max()),
        "P95_abs_error_m": float(np.percentile(ae, 95)),
        "bias_m": float(err.mean()),  # >0: el modelo mide de más
        "mean_relative_error_percent": float((ae / gt).mean() * 100.0),
    }


def evaluate(rows: list[dict], max_validation_error_m: float = 0.05, min_refs_for_p95: int = 5) -> dict:
    """rows: lista de {name, role, ground_truth_m, model_m, ...}."""
    detailed = []
    for r in rows:
        err = float(r["model_m"]) - float(r["ground_truth_m"])
        detailed.append({
            **r,
            "error_m": err,
            "abs_error_m": abs(err),
            "relative_error_percent": abs(err) / float(r["ground_truth_m"]) * 100.0,
        })

    cal = [r for r in detailed if r["role"] == "calibrate"]
    val = [r for r in detailed if r["role"] == "validate"]
    warnings: list[str] = []

    if not val:
        warnings.append(
            "No hay referencias con role: validate. Los errores de las referencias de calibración "
            "no miden la exactitud real del modelo (el ajuste los minimiza a propósito)."
        )
    else:
        if len(val) < min_refs_for_p95:
            warnings.append(
                f"Solo hay {len(val)} referencia(s) de validación: el P95 y el RMSE no son representativos "
                f"(se recomiendan al menos {min_refs_for_p95})."
            )
        worst = max(val, key=lambda r: r["abs_error_m"])
        if worst["abs_error_m"] > max_validation_error_m:
            warnings.append(
                f"La referencia de validación '{worst['name']}' tiene error {worst['abs_error_m'] * 1000:.1f} mm, "
                f"sobre el umbral configurado ({max_validation_error_m * 1000:.0f} mm)."
            )

    return {
        "measurements": detailed,
        "summary_validation": _stats(val),
        "summary_calibration": _stats(cal),
        "warnings": warnings,
        "note": "Los estadísticos de validación usan solo referencias que no participaron en el ajuste de escala.",
    }
