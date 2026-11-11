"""2. DOWNSAMPLING + 3. OUTLIER FILTERING."""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def _stage(n_in: int, n_out: int) -> dict:
    return {
        "input_points": n_in,
        "output_points": n_out,
        "removed_points": n_in - n_out,
        "removed_percent": _pct(n_in - n_out, n_in),
    }


def drop_invalid_points(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Devuelve una copia sin puntos NaN/inf."""
    mask = np.isfinite(np.asarray(pcd.points)).all(axis=1)
    if mask.all():
        return o3d.geometry.PointCloud(pcd)
    return pcd.select_by_index(np.flatnonzero(mask).tolist())


def statistical_outlier_removal(pcd, nb_neighbors: int, std_ratio: float):
    """Quita puntos cuya distancia media a sus vecinos es anómala."""
    kept, ind = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
    )
    return kept, pcd.select_by_index(ind, invert=True)


def radius_outlier_removal(pcd, nb_points: int, radius: float):
    """Quita puntos con pocos vecinos dentro de un radio."""
    kept, ind = pcd.remove_radius_outlier(nb_points=int(nb_points), radius=float(radius))
    return kept, pcd.select_by_index(ind, invert=True)


def run_cleaning(pcd, cfg: dict, params: dict):
    """Ejecuta limpieza completa.

    Devuelve (nube_limpia, nube_de_outliers_removidos, reporte).
    `params` trae las longitudes ya resueltas a unidades de la nube.
    """
    n_raw = len(pcd.points)
    work = drop_invalid_points(pcd)
    n_valid = len(work.points)
    report: dict = {"invalid_points_dropped": _stage(n_raw, n_valid)}

    if cfg["downsample"]["enabled"]:
        work = work.voxel_down_sample(params["voxel_size"])
        report["downsample"] = {
            "enabled": True,
            "voxel_size_used": params["voxel_size"],
            **_stage(n_valid, len(work.points)),
        }
    else:
        report["downsample"] = {"enabled": False}

    n_in = len(work.points)
    if n_in == 0:
        raise ValueError("No quedan puntos después del downsampling; reduce voxel_size.")

    st, rd = cfg["outlier"]["statistical"], cfg["outlier"]["radius"]
    s_kept, s_removed = statistical_outlier_removal(work, st["nb_neighbors"], st["std_ratio"])
    r_kept, r_removed = radius_outlier_removal(work, rd["nb_points"], params["outlier_radius"])

    comparison = {
        "statistical": {
            "parameters": {"nb_neighbors": st["nb_neighbors"], "std_ratio": st["std_ratio"]},
            **_stage(n_in, len(s_kept.points)),
        },
        "radius": {
            "parameters": {"nb_points": rd["nb_points"], "radius_used": params["outlier_radius"]},
            **_stage(n_in, len(r_kept.points)),
        },
    }
    selected = cfg["outlier"]["selected"]
    kept, removed = (s_kept, s_removed) if selected == "statistical" else (r_kept, r_removed)

    report["outlier_comparison"] = comparison
    report["selected_method"] = selected
    report["outlier_removed_percent"] = comparison[selected]["removed_percent"]
    report["overall"] = _stage(n_raw, len(kept.points))
    log.info(
        "Limpieza: %d -> %d puntos (outliers %s: %.2f%%)",
        n_raw, len(kept.points), selected, report["outlier_removed_percent"],
    )
    return kept, removed, report
