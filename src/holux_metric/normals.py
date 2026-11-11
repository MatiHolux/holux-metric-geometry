"""4. NORMALS: estimar (Nx, Ny, Nz) por punto."""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


def compute_normals(pcd, radius: float, max_nn: int, orient: bool, orient_k: int) -> dict:
    """Estima normals in-place y devuelve un resumen para qa.json."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn))
    )
    info: dict = {"radius_used": float(radius), "max_nn": int(max_nn), "orientation": "none"}
    warnings: list[str] = []

    if orient:
        try:
            pcd.orient_normals_consistent_tangent_plane(int(orient_k))
            info["orientation"] = "consistent_tangent_plane"
        except Exception as exc:  # puede fallar con nubes muy ralas o desconectadas
            info["orientation"] = "failed"
            warnings.append(f"No se pudo orientar las normals de forma consistente: {exc}")

    normals = np.asarray(pcd.normals)
    info["normals_finite"] = bool(np.isfinite(normals).all())
    info["warnings"] = warnings
    return info
