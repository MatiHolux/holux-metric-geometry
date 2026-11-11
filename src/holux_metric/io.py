"""Entrada/salida: nubes de puntos, JSON y checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


class InputError(Exception):
    """El archivo de entrada no existe, no se puede leer o está vacío."""


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"No existe el archivo: {p}")
    pcd = o3d.io.read_point_cloud(str(p))
    if pcd.is_empty():
        raise InputError(f"La nube está vacía o el formato no es legible: {p}")
    return pcd


def save_point_cloud(path: str | Path, pcd: o3d.geometry.PointCloud) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(p), pcd, write_ascii=False):
        raise OSError(f"No se pudo escribir la nube: {p}")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"No serializable a JSON: {type(obj)}")


def write_json(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
