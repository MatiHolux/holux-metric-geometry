"""6. QA: resultado cuantificable y reproducible."""
from __future__ import annotations

import platform
from datetime import datetime, timezone

import numpy as np
import open3d as o3d

from holux_metric.inspect import bbox_from_points


def environment_info() -> dict:
    return {"Open3D_version": o3d.__version__, "Python_version": platform.python_version()}


def build_qa(inspect_data: dict, cleaning: dict, normals_info: dict, planes: list,
             cleaned, cfg: dict, cfg_hash: str, params: dict, input_sha256: str,
             extra_warnings: list[str]) -> dict:
    warnings = list(extra_warnings) + list(normals_info.get("warnings", []))
    out_pts = np.asarray(cleaned.points)
    finite_out = bool(np.isfinite(out_pts).all()) if out_pts.size else False
    overall = cleaning["overall"]
    outlier_pct = cleaning["outlier_removed_percent"]

    if inspect_data["invalid_points"] > 0:
        warnings.append(f"{inspect_data['invalid_points']} puntos no finitos fueron descartados.")
    if outlier_pct > cfg["qa"]["max_removed_percent"]:
        warnings.append(
            f"El filtro de outliers quitó {outlier_pct:.2f}% (umbral {cfg['qa']['max_removed_percent']}%)."
        )
    if overall["output_points"] < cfg["qa"]["min_output_points"]:
        warnings.append(f"Quedan solo {overall['output_points']} puntos (mínimo {cfg['qa']['min_output_points']}).")
    if not planes:
        warnings.append("No se detectó ningún plano; revisa distance_threshold y min_points.")
    if inspect_data["has_colors"] and not cleaned.has_colors():
        warnings.append("La nube de entrada tenía color y la de salida no.")
    if not normals_info.get("normals_finite", False):
        warnings.append("Hay normals no finitas en la salida.")

    failed = overall["output_points"] == 0 or not finite_out
    status = "FAIL" if failed else ("WARN" if warnings else "PASS")

    return {
        "status": status,
        "input_readable": True,
        "colors_preserved": (not inspect_data["has_colors"]) or bool(cleaned.has_colors()),
        "finite_coordinates": finite_out,
        "input_points": overall["input_points"],
        "output_points": overall["output_points"],
        "removed_points": overall["removed_points"],
        "removed_percent": overall["removed_percent"],
        "outlier_removed_percent": outlier_pct,
        "bounding_box_before": inspect_data["bounding_box"],
        "bounding_box_after": bbox_from_points(out_pts),
        "planes_detected": len(planes),
        "normals": {k: v for k, v in normals_info.items() if k != "warnings"},
        "parameters_resolved": params,
        "warnings": warnings,
        "algorithm_version": cfg["algorithm_version"],
        "Open3D_version": o3d.__version__,
        "Python_version": platform.python_version(),
        "config_sha256": cfg_hash,
        "input_sha256": input_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_failed_qa(error: str, cfg: dict, cfg_hash: str) -> dict:
    return {
        "status": "FAIL",
        "input_readable": False,
        "error": error,
        "warnings": [error],
        "algorithm_version": cfg["algorithm_version"],
        **environment_info(),
        "config_sha256": cfg_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
