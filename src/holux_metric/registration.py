"""Etapa siguiente: REGISTRATION (FPFH + RANSAC global + ICP). Aún no implementado."""
from __future__ import annotations


def register(source, target, config: dict) -> dict:
    """Debe devolver el contenido de registration.json:
    source, target, transformation (4x4), fitness, inlier_rmse,
    correspondence_count, status.
    """
    raise NotImplementedError("Registration se implementa después de cerrar O3D-1")
