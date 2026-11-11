"""Herramienta para elegir puntos de referencia en una nube (ventana de Open3D).

Uso: holux-metric pick nube.ply
  Shift + clic izquierdo = elegir punto     Q = terminar
Genera picked_points.json y una plantilla de referencias (pares consecutivos de puntos).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

TEMPLATE_HEADER = """\
# Plantilla generada por 'holux-metric pick'.
# 1) Completa real_distance_m (en metros) con la medida física de cada par de puntos.
# 2) Deja role: calibrate en las que usarás para ajustar la escala y pon role: validate
#    en las que reservas para comprobar la exactitud (no participan en el ajuste).
# 3) Opcional: frame.floor_plane_id (p. ej. plane_001, requiere --planes), frame.x_axis y frame.origin_visual.
"""


def pick_points(input_path, output_json, template_yaml) -> dict:
    import open3d as o3d

    from holux_metric.io import load_point_cloud, write_json

    pcd = load_point_cloud(input_path)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="holux-metric pick | Shift+clic = elegir punto | Q = terminar")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3.0
    vis.run()
    vis.destroy_window()
    idx = list(vis.get_picked_points())

    pts = np.asarray(pcd.points)[idx] if idx else np.zeros((0, 3))
    picked = [{"id": f"p{i + 1:02d}", "index": int(j), "xyz": pts[i].tolist()} for i, j in enumerate(idx)]
    write_json(output_json, {"source": str(input_path), "points": picked})

    pairs = []
    for k in range(0, len(picked) - 1, 2):
        a, b = picked[k], picked[k + 1]
        pairs.append({
            "name": f"ref_{k // 2 + 1:02d}",
            "point_a": a["xyz"],
            "point_b": b["xyz"],
            "real_distance_m": None,
            "role": "calibrate",
            "distance_in_cloud_units": float(np.linalg.norm(np.array(b["xyz"]) - np.array(a["xyz"]))),
        })
    doc = {
        "site": "site001",
        "references": pairs,
        "frame": {"floor_plane_id": None, "x_axis": None, "origin_visual": None},
    }
    tp = Path(template_yaml)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(TEMPLATE_HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"picked": len(picked), "pairs": len(pairs), "template": str(tp), "points_json": str(output_json)}
