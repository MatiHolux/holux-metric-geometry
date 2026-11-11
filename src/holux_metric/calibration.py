"""SITE CALIBRATION: coordenadas visuales -> coordenadas del Site (metros).

Modelo:    P_site = s * R * P_visual + t

  s  escala (se estima con distancias reales conocidas)
  R  rotación (Z hacia arriba, usando el plano del piso; X opcional)
  t  traslación (origen del Site)

Las funciones de este módulo son numpy puro (sin Open3D) salvo run_calibration,
que importa Open3D solo cuando se llama.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from holux_metric.measurements import evaluate

log = logging.getLogger(__name__)

ALGORITHM_VERSION = "calib-1.0.0"
ROLES = ("calibrate", "validate")


class CalibrationError(ValueError):
    """Referencias o parámetros de calibración inválidos."""


# ---------------------------------------------------------------- parsing

def _vec3(value, name: str) -> np.ndarray:
    try:
        v = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise CalibrationError(f"{name} debe ser [x, y, z] con números") from None
    if v.shape != (3,) or not np.isfinite(v).all():
        raise CalibrationError(f"{name} debe ser [x, y, z] con números finitos (valor: {value!r})")
    return v


def parse_references(doc: dict) -> tuple[list[dict], dict]:
    """Valida el documento de referencias. Devuelve (referencias, frame)."""
    if not isinstance(doc, dict) or "references" not in doc:
        raise CalibrationError("El archivo de referencias debe tener la clave 'references'")
    raw_list = doc["references"] or []
    refs: list[dict] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            raise CalibrationError(f"references[{i}] debe ser un mapa con name, point_a, point_b, ...")
        name = str(raw.get("name", f"ref_{i + 1:02d}"))
        if name in seen:
            raise CalibrationError(f"Nombre de referencia repetido: {name}")
        seen.add(name)
        a = _vec3(raw.get("point_a"), f"{name}.point_a")
        b = _vec3(raw.get("point_b"), f"{name}.point_b")
        real = raw.get("real_distance_m")
        if isinstance(real, bool) or not isinstance(real, (int, float)) or not np.isfinite(real) or real <= 0:
            raise CalibrationError(f"{name}: real_distance_m debe ser un número > 0 (¿falta completarlo?)")
        role = raw.get("role", "calibrate")
        if role not in ROLES:
            raise CalibrationError(f"{name}: role debe ser 'calibrate' o 'validate' (valor: {role!r})")
        if np.linalg.norm(b - a) <= 1e-12:
            raise CalibrationError(f"{name}: point_a y point_b son el mismo punto")
        refs.append({"name": name, "role": role, "point_a": a, "point_b": b, "real_distance_m": float(real)})
    if not refs:
        raise CalibrationError("No hay referencias en el archivo")
    return refs, (doc.get("frame") or {})


# ------------------------------------------------------------------ scale

def estimate_scale(refs: list[dict], warn_cv_percent: float):
    """Escala por mínimos cuadrados con las referencias role=calibrate.

    s = sum(d_model * d_real) / sum(d_model^2)
    """
    cal = [r for r in refs if r["role"] == "calibrate"]
    if not cal:
        raise CalibrationError("Se necesita al menos una referencia con role: calibrate")
    dm = np.array([np.linalg.norm(r["point_b"] - r["point_a"]) for r in cal])
    dr = np.array([r["real_distance_m"] for r in cal])
    s = float(np.dot(dm, dr) / np.dot(dm, dm))
    ratios = dr / dm
    cv = float(np.std(ratios) / np.mean(ratios) * 100.0) if len(cal) > 1 else None

    warnings: list[str] = []
    if len(cal) == 1:
        warnings.append(
            "Solo hay una referencia de calibración: la escala queda exacta para esa medida pero "
            "no hay redundancia para detectar errores. Usa al menos 2-3."
        )
    if cv is not None and cv > warn_cv_percent:
        warnings.append(
            f"Las referencias de calibración no son consistentes entre sí (la escala varía {cv:.2f}% "
            f"entre ellas; umbral {warn_cv_percent}%). Revisa los puntos elegidos y las medidas reales."
        )
    diag = {
        "n_calibration_refs": len(cal),
        "scale_ratios_per_reference": {r["name"]: float(x) for r, x in zip(cal, ratios)},
        "ratio_cv_percent": cv,
    }
    return s, diag, warnings


# ------------------------------------------------------------------ frame

def rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotación mínima que lleva el vector a sobre el vector b (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:  # vectores opuestos: giro de 180° sobre un eje perpendicular
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) / (1.0 + c)


def _find_plane(planes_doc: dict | None, plane_id: str) -> dict:
    if planes_doc is None:
        raise CalibrationError("frame.floor_plane_id requiere el archivo planes.json (opción --planes)")
    for p in planes_doc.get("planes", []):
        if p.get("id") == plane_id:
            return p
    available = [p.get("id") for p in planes_doc.get("planes", [])]
    raise CalibrationError(f"No existe el plano '{plane_id}' en planes.json (disponibles: {available})")


def build_frame(s: float, points: np.ndarray, frame: dict, plane: dict | None):
    """Construye R y t para fijar Z arriba, X opcional y origen del Site."""
    warnings: list[str] = []
    median = np.median(points, axis=0)
    R = np.eye(3)
    floor_pt = None
    info: dict = {"floor_plane_id": None, "floor_normal_flipped": False, "x_axis_defined": False}

    if plane is not None:
        n = np.asarray(plane["normal"], float)
        n = n / np.linalg.norm(n)
        d = float(plane["offset_d"])
        if np.dot(n, median) + d < 0:  # la normal debe apuntar hacia la escena
            n, d = -n, -d
            info["floor_normal_flipped"] = True
        R = rotation_from_to(n, np.array([0.0, 0.0, 1.0]))
        floor_pt = -d * n
        info["floor_plane_id"] = plane["id"]
        # Los puntos del propio piso quedan a ambos lados del plano por el ruido: se tolera un 2% de la
        # extensión robusta de la escena (percentiles 1-99) antes de considerar que un punto está "bajo" el plano.
        robust_diag = float(np.linalg.norm(np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)))
        above = float(np.mean(points @ n + d > -0.02 * robust_diag))
        if above < 0.9:
            warnings.append(
                f"Solo {above * 100:.0f}% de los puntos queda sobre el plano {plane['id']}; un piso debería "
                "dejar casi toda la escena de un lado. Revisa que sea el piso (mira planes.png)."
            )
    else:
        warnings.append(
            "No se definió el piso (frame.floor_plane_id): el eje Z del Site no está garantizado como vertical."
        )

    x_axis = frame.get("x_axis")
    if x_axis:
        v = _vec3(x_axis.get("to"), "frame.x_axis.to") - _vec3(x_axis.get("from"), "frame.x_axis.from")
        vr = R @ v
        if np.hypot(vr[0], vr[1]) < 1e-9:
            raise CalibrationError("frame.x_axis es vertical: elige dos puntos a lo largo de una pared o del piso")
        phi = -np.arctan2(vr[1], vr[0])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ R
        info["x_axis_defined"] = True
        if plane is None:
            warnings.append("frame.x_axis se interpretó respecto al eje Z de la nube visual (no hay piso definido).")

    origin_given = frame.get("origin_visual") is not None
    origin = _vec3(frame["origin_visual"], "frame.origin_visual") if origin_given else median
    info["origin_visual"] = origin.tolist()
    info["origin_is_cloud_median"] = not origin_given

    o_pre = s * (R @ origin)
    if floor_pt is not None:
        z_floor = float((s * (R @ floor_pt))[2])
        t = np.array([-o_pre[0], -o_pre[1], -z_floor])  # origen en XY del punto, Z = piso
    else:
        t = -o_pre
    info["axes"] = {
        "Z": "arriba (normal del piso)" if plane is not None else "no definido",
        "X": "definido por frame.x_axis" if info["x_axis_defined"] else "arbitrario (no definido)",
        "Y": "Z x X (sistema de mano derecha)",
    }
    return R, t, info, warnings


def apply_transform(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * (np.asarray(points, float) @ R.T) + t


def calibrate_points(points: np.ndarray, references_doc: dict, planes_doc: dict | None, cal_cfg: dict):
    """Estima s, R, t y evalúa las referencias. Devuelve (resultado, (s, R, t), evaluación)."""
    pts = np.asarray(points, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        raise CalibrationError("La nube no tiene puntos finitos")

    refs, frame = parse_references(references_doc)
    s, scale_diag, warnings = estimate_scale(refs, cal_cfg["warn_scale_cv_percent"])

    plane = _find_plane(planes_doc, frame["floor_plane_id"]) if frame.get("floor_plane_id") else None
    R, t, frame_info, frame_warnings = build_frame(s, pts, frame, plane)
    warnings += frame_warnings

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise CalibrationError("La rotación calculada no es válida (det != 1)")

    rows = []
    for r in refs:
        qa = apply_transform(r["point_a"][None, :], s, R, t)[0]
        qb = apply_transform(r["point_b"][None, :], s, R, t)[0]
        rows.append({
            "name": r["name"],
            "role": r["role"],
            "ground_truth_m": r["real_distance_m"],
            "model_m": float(np.linalg.norm(qb - qa)),
            "distance_in_cloud_units": float(np.linalg.norm(r["point_b"] - r["point_a"])),
        })
    evaluation = evaluate(rows, cal_cfg["max_validation_error_m"], cal_cfg["min_validation_refs_for_p95"])
    warnings += evaluation["warnings"]

    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    result = {
        "status": "WARN" if warnings else "PASS",
        "formula": "P_site = s * R * P_visual + t",
        "units": "meters",
        "scale": s,
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "matrix_4x4": M.tolist(),
        "frame": frame_info,
        "scale_diagnostics": scale_diag,
        "references_used": [{"name": r["name"], "role": r["role"]} for r in refs],
        "warnings": warnings,
        "algorithm_version": ALGORITHM_VERSION,
    }
    return result, (s, R, t), evaluation


# ------------------------------------------------------------- file level

def run_calibration(input_path, references_path, planes_path, cfg: dict, output_dir) -> dict:
    """Calibra una nube .ply y escribe site_reference_metric.ply, site_calibration.json y measurements.json."""
    import open3d as o3d  # import tardío: el resto del módulo no depende de Open3D

    from holux_metric.inspect import bbox_from_points
    from holux_metric.io import load_point_cloud, save_point_cloud, sha256_file, write_json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_path = Path(references_path)
    if not ref_path.is_file():
        raise CalibrationError(f"No existe el archivo de referencias: {ref_path}")
    with ref_path.open("r", encoding="utf-8") as fh:
        references_doc = yaml.safe_load(fh)

    planes_doc = None
    if planes_path:
        pp = Path(planes_path)
        if not pp.is_file():
            raise CalibrationError(f"No existe planes.json: {pp}")
        planes_doc = json.loads(pp.read_text(encoding="utf-8"))

    pcd = load_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pcd = pcd.select_by_index(np.flatnonzero(finite).tolist())
        pts = np.asarray(pcd.points)

    result, (s, R, t), evaluation = calibrate_points(pts, references_doc, planes_doc, cfg["calibration"])

    pcd.points = o3d.utility.Vector3dVector(apply_transform(pts, s, R, t))
    if pcd.has_normals():
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) @ R.T)  # las normales solo rotan
    save_point_cloud(out / "site_reference_metric.ply", pcd)

    site_pts = np.asarray(pcd.points)
    p1, p99 = np.percentile(site_pts, 1, axis=0), np.percentile(site_pts, 99, axis=0)
    result.update({
        "input": {
            "cloud": str(input_path),
            "cloud_sha256": sha256_file(input_path),
            "references": str(ref_path),
            "references_sha256": sha256_file(ref_path),
            "planes": str(planes_path) if planes_path else None,
        },
        "bounding_box_site": bbox_from_points(site_pts),
        "extent_p1_p99_m": (p99 - p1).tolist(),
        "Open3D_version": o3d.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    write_json(out / "site_calibration.json", result)
    write_json(out / "measurements.json", evaluation)
    log.info("Calibración: escala s=%.6f, status %s", s, result["status"])
    return {"calibration": result, "measurements": evaluation}
