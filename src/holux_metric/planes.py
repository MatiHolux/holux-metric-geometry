"""5. PLANES: planos dominantes por RANSAC iterativo."""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from holux_metric.inspect import bbox_from_points

log = logging.getLogger(__name__)


def detect_planes(pcd, cfg_planes: dict, distance_threshold: float, seed: int | None = None):
    """Detecta hasta `max_planes` planos sacando los inliers en cada iteración.

    Devuelve (lista_de_planos, labels) donde labels[i] es el índice del plano
    del punto i, o -1 si no pertenece a ninguno.
    """
    if seed is not None:
        try:
            o3d.utility.random.seed(int(seed))
        except AttributeError:
            log.warning("Esta versión de Open3D no permite fijar la semilla aleatoria.")

    n = len(pcd.points)
    pts = np.asarray(pcd.points)
    labels = np.full(n, -1, dtype=int)
    remaining = np.arange(n)
    planes: list[dict] = []

    for k in range(int(cfg_planes["max_planes"])):
        if remaining.size < cfg_planes["min_points"]:
            break
        sub = pcd.select_by_index(remaining.tolist())
        model, inliers = sub.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=int(cfg_planes["ransac_n"]),
            num_iterations=int(cfg_planes["num_iterations"]),
        )
        if len(inliers) < cfg_planes["min_points"]:
            break

        a, b, c, d = (float(v) for v in model)
        norm = float(np.linalg.norm([a, b, c]))
        global_idx = remaining[np.asarray(inliers)]
        labels[global_idx] = k
        inlier_pts = pts[global_idx]
        planes.append(
            {
                "id": f"plane_{k + 1:03d}",
                "normal": [a / norm, b / norm, c / norm],
                "offset_d": d / norm,
                "centroid": inlier_pts.mean(axis=0).tolist(),
                "point_count": int(global_idx.size),
                "point_fraction": round(float(global_idx.size) / n, 6),
                "inlier_bbox": bbox_from_points(inlier_pts),
            }
        )
        remaining = np.delete(remaining, np.asarray(inliers))

    log.info("Planos detectados: %d", len(planes))
    return planes, labels
