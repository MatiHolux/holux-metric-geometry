"""Orquestación de O3D-1: LOAD -> INSPECT -> CLEAN -> NORMALS -> PLANES -> QA."""
from __future__ import annotations

import logging
from pathlib import Path

from holux_metric.clean import run_cleaning
from holux_metric.config import config_sha256, resolve_parameters
from holux_metric.inspect import describe_source, inspect_point_cloud
from holux_metric.io import InputError, load_point_cloud, save_point_cloud, write_json
from holux_metric.normals import compute_normals
from holux_metric.planes import detect_planes
from holux_metric.qa import build_failed_qa, build_qa

log = logging.getLogger(__name__)


def run_o3d1(input_path: str | Path, cfg: dict, output_dir: str | Path, make_preview: bool = True) -> dict:
    """Ejecuta O3D-1 y devuelve el contenido de qa.json (también lo escribe a disco)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_sha256(cfg)

    # LOAD
    try:
        raw = load_point_cloud(input_path)
    except InputError as exc:
        qa = build_failed_qa(str(exc), cfg, cfg_hash)
        write_json(out / "qa.json", qa)
        return qa

    # INSPECT
    inspect_data = inspect_point_cloud(raw)
    source = describe_source(input_path)
    write_json(out / "inspect.json", {**source, **inspect_data})
    bbox = inspect_data["bounding_box"]
    if bbox is None or bbox["diagonal"] <= 0:
        qa = build_failed_qa("La nube no tiene puntos finitos o tiene extensión cero.", cfg, cfg_hash)
        write_json(out / "qa.json", qa)
        return qa
    params = resolve_parameters(cfg, bbox["diagonal"])

    # DOWNSAMPLE + OUTLIER FILTER
    cleaned, removed, cleaning = run_cleaning(raw, cfg, params)
    write_json(out / "cleaning.json", {"parameters_resolved": params, **cleaning})

    # NORMALS
    n = cfg["normals"]
    normals_info = compute_normals(cleaned, params["normals_radius"], n["max_nn"], n["orient"], n["orient_k"])

    # PLANES
    planes, labels = detect_planes(cleaned, cfg["planes"], params["plane_distance_threshold"], cfg["seed"])
    write_json(out / "planes.json", {
        "parameters": {**cfg["planes"], "distance_threshold_used": params["plane_distance_threshold"]},
        "planes_detected": len(planes),
        "unassigned_points": int((labels < 0).sum()),
        "planes": planes,
    })

    # OUTPUT PLY
    save_point_cloud(out / "visual_reference_clean.ply", cleaned)

    # PREVIEW
    extra_warnings: list[str] = []
    if make_preview and cfg["preview"]["enabled"]:
        try:
            from holux_metric.preview import generate_previews
            generate_previews(out / "preview", raw, cleaned, removed, labels, cfg["preview"])
        except Exception as exc:  # las imágenes nunca deben botar el pipeline
            extra_warnings.append(f"No se pudieron generar las previews: {exc}")

    # QA
    qa = build_qa(inspect_data, cleaning, normals_info, planes, cleaned, cfg, cfg_hash,
                  params, source["sha256"], extra_warnings)
    write_json(out / "qa.json", qa)
    log.info("QA: %s (%d warnings)", qa["status"], len(qa["warnings"]))
    return qa
