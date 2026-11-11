"""1. INSPECTION: qué contiene exactamente la nube."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from holux_metric.io import sha256_file


def bbox_from_points(points: np.ndarray) -> dict | None:
    """Bounding box alineado a ejes de un arreglo Nx3 (se asume finito)."""
    if points.shape[0] == 0:
        return None
    mn, mx = points.min(axis=0), points.max(axis=0)
    extent = mx - mn
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "extent": extent.tolist(),
        "center": ((mn + mx) / 2).tolist(),
        "diagonal": float(np.linalg.norm(extent)),
    }


def inspect_point_cloud(pcd: o3d.geometry.PointCloud) -> dict:
    pts = np.asarray(pcd.points)
    finite = np.isfinite(pts).all(axis=1)
    valid = int(finite.sum())
    info = {
        "num_points": int(pts.shape[0]),
        "valid_points": valid,
        "invalid_points": int(pts.shape[0] - valid),
        "has_colors": bool(pcd.has_colors()),
        "has_normals": bool(pcd.has_normals()),
        "bounding_box": bbox_from_points(pts[finite]),  # solo puntos válidos
    }
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        info["color_range"] = {"min": float(cols.min()), "max": float(cols.max())}
    return info


def describe_source(path: str | Path) -> dict:
    p = Path(path)
    return {
        "source": str(p),
        "file_size_bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }
