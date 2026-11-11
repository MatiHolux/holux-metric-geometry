"""Imágenes de previsualización (matplotlib, sin necesidad de ventana ni GPU).

Cada imagen muestra tres proyecciones: XY (planta), XZ y YZ (alzados).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

_VIEWS = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
GREY = np.array([0.6, 0.6, 0.6])
RED = np.array([0.85, 0.1, 0.1])


def _clean(points: np.ndarray, colors: np.ndarray | None):
    mask = np.isfinite(points).all(axis=1)
    return points[mask], (colors[mask] if colors is not None else None)


def _subsample(points, colors, max_points, rng):
    if points.shape[0] <= max_points:
        return points, colors
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], (colors[idx] if colors is not None else None)


def _render(path: Path, layers: list[tuple], title: str, pv: dict) -> None:
    """layers: lista de (puntos Nx3, colores Nx3 o None)."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j, li, lj) in zip(axes, _VIEWS):
        for pts, cols in layers:
            pts, cols = _clean(pts, cols)
            if pts.shape[0] == 0:
                continue
            pts, cols = _subsample(pts, cols, pv["max_points"], rng)
            ax.scatter(pts[:, i], pts[:, j], s=pv["point_size"], c=cols if cols is not None else GREY[None, :],
                       linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_title(f"{li}{lj}")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=pv["dpi"])
    plt.close(fig)


def _colors_of(pcd) -> np.ndarray | None:
    return np.asarray(pcd.colors) if pcd.has_colors() else None


def generate_previews(out_dir: str | Path, original, cleaned, removed, labels: np.ndarray, pv: dict) -> list[str]:
    out = Path(out_dir)
    cleaned_pts = np.asarray(cleaned.points)
    paths = []

    p = out / "input.png"
    _render(p, [(np.asarray(original.points), _colors_of(original))], "Entrada (sparse_pc.ply)", pv)
    paths.append(str(p))

    p = out / "cleaned.png"
    _render(p, [(cleaned_pts, _colors_of(cleaned))], "Nube limpia", pv)
    paths.append(str(p))

    p = out / "outliers.png"
    removed_pts = np.asarray(removed.points)
    red = np.tile(RED, (removed_pts.shape[0], 1))
    _render(p, [(cleaned_pts, None), (removed_pts, red)], "Outliers removidos (rojo) sobre nube limpia (gris)", pv)
    paths.append(str(p))

    p = out / "normals.png"
    normals_rgb = (np.asarray(cleaned.normals) + 1.0) / 2.0 if cleaned.has_normals() else None
    _render(p, [(cleaned_pts, normals_rgb)], "Normals (color = dirección)", pv)
    paths.append(str(p))

    p = out / "planes.png"
    cmap = plt.get_cmap("tab10")
    colors = np.tile(GREY, (cleaned_pts.shape[0], 1))
    for k in range(int(labels.max()) + 1 if labels.size else 0):
        colors[labels == k] = cmap(k % 10)[:3]
    _render(p, [(cleaned_pts, colors)], "Planos detectados (gris = sin plano)", pv)
    paths.append(str(p))
    return paths
