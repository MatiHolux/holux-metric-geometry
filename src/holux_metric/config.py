"""Carga, validación y resolución de parámetros del pipeline (YAML)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict = {
    "algorithm_version": "o3d-1.0.0",
    "seed": 42,
    "length_mode": "relative",
    "downsample": {"enabled": True, "voxel_size": 0.005},
    "outlier": {
        "selected": "statistical",
        "statistical": {"nb_neighbors": 20, "std_ratio": 2.0},
        "radius": {"nb_points": 8, "radius": 0.02},
    },
    "normals": {"radius": 0.02, "max_nn": 30, "orient": True, "orient_k": 15},
    "planes": {
        "max_planes": 6,
        "distance_threshold": 0.005,
        "ransac_n": 3,
        "num_iterations": 2000,
        "min_points": 200,
    },
    "qa": {"max_removed_percent": 30.0, "min_output_points": 1000},
    "preview": {"enabled": True, "max_points": 150000, "point_size": 0.5, "dpi": 110},
    "calibration": {
        "warn_scale_cv_percent": 2.0,      # aviso si las referencias dan escalas muy distintas entre sí
        "max_validation_error_m": 0.05,    # aviso si una referencia de validación supera este error
        "min_validation_refs_for_p95": 5,  # mínimo de validaciones para que P95/RMSE sean representativos
    },
}


class ConfigError(ValueError):
    """Configuración inválida."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def validate(cfg: dict) -> None:
    if cfg["length_mode"] not in ("relative", "absolute"):
        raise ConfigError("length_mode debe ser 'relative' o 'absolute'")
    if cfg["outlier"]["selected"] not in ("statistical", "radius"):
        raise ConfigError("outlier.selected debe ser 'statistical' o 'radius'")
    positive = [
        ("downsample.voxel_size", cfg["downsample"]["voxel_size"]),
        ("outlier.radius.radius", cfg["outlier"]["radius"]["radius"]),
        ("outlier.statistical.std_ratio", cfg["outlier"]["statistical"]["std_ratio"]),
        ("normals.radius", cfg["normals"]["radius"]),
        ("planes.distance_threshold", cfg["planes"]["distance_threshold"]),
    ]
    for name, value in positive:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{name} debe ser un número > 0 (valor: {value!r})")
    if cfg["planes"]["ransac_n"] < 3:
        raise ConfigError("planes.ransac_n debe ser >= 3")
    for name in ("warn_scale_cv_percent", "max_validation_error_m"):
        value = cfg["calibration"][name]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"calibration.{name} debe ser un número > 0 (valor: {value!r})")


def load_config(path: str | Path | None = None) -> dict:
    """Carga el YAML sobre los valores por defecto y valida el resultado."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"No existe el archivo de configuración: {p}")
        with p.open("r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ConfigError("El YAML de configuración debe ser un mapa clave: valor")
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ConfigError(f"Claves desconocidas en la configuración: {sorted(unknown)}")
        cfg = _deep_merge(cfg, user)
    validate(cfg)
    return cfg


def config_sha256(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_parameters(cfg: dict, diagonal: float) -> dict[str, Any]:
    """Convierte las longitudes del config a unidades de la nube."""
    relative = cfg["length_mode"] == "relative"

    def length(value: float) -> float:
        return float(value) * diagonal if relative else float(value)

    return {
        "length_mode": cfg["length_mode"],
        "bbox_diagonal": diagonal,
        "voxel_size": length(cfg["downsample"]["voxel_size"]) if cfg["downsample"]["enabled"] else None,
        "outlier_radius": length(cfg["outlier"]["radius"]["radius"]),
        "normals_radius": length(cfg["normals"]["radius"]),
        "plane_distance_threshold": length(cfg["planes"]["distance_threshold"]),
    }
