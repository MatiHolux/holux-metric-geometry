"""Línea de comandos: holux-metric inspect | clean | pick | calibrate"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from holux_metric.calibration import CalibrationError
from holux_metric.config import ConfigError, load_config
from holux_metric.inspect import describe_source, inspect_point_cloud
from holux_metric.io import InputError, load_point_cloud, write_json
from holux_metric.pipeline import run_o3d1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="holux-metric", description="HO-LUX Spatial - geometría métrica")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspecciona una nube y muestra inspect.json")
    p_inspect.add_argument("input", help="Ruta al .ply")
    p_inspect.add_argument("-o", "--output", help="Si se indica, guarda el JSON en esta ruta")

    p_clean = sub.add_parser("clean", help="Ejecuta O3D-1 completo (limpieza, normals, planos, QA)")
    p_clean.add_argument("input", help="Ruta al .ply")
    p_clean.add_argument("-c", "--config", help="YAML de configuración (por defecto: valores internos)")
    p_clean.add_argument("-o", "--output", default="output/o3d-1", help="Carpeta de salida")
    p_clean.add_argument("--no-preview", action="store_true", help="No generar imágenes")

    p_pick = sub.add_parser("pick", help="Elegir puntos de referencia en una ventana 3D y generar la plantilla de referencias")
    p_pick.add_argument("input", help="Ruta al .ply")
    p_pick.add_argument("-o", "--output", default="output/calibration/picked_points.json")
    p_pick.add_argument("-t", "--template", default="output/calibration/references.template.yaml")

    p_cal = sub.add_parser("calibrate", help="Calibra la nube a coordenadas del Site (metros)")
    p_cal.add_argument("input", help="Ruta al .ply (idealmente visual_reference_clean.ply)")
    p_cal.add_argument("-r", "--references", required=True, help="YAML con las referencias (distancias reales)")
    p_cal.add_argument("-p", "--planes", help="planes.json de O3D-1 (necesario si usas frame.floor_plane_id)")
    p_cal.add_argument("-c", "--config", help="YAML de configuración")
    p_cal.add_argument("-o", "--output", default="output/calibration", help="Carpeta de salida")
    return parser


def _fmt_mm(value) -> str:
    return "n/d" if value is None else f"{value * 1000:.1f} mm"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        if args.command == "inspect":
            data = {**describe_source(args.input), **inspect_point_cloud(load_point_cloud(args.input))}
            print(json.dumps(data, indent=2, ensure_ascii=False))
            if args.output:
                write_json(args.output, data)
            return 0

        if args.command == "pick":
            from holux_metric.pick import pick_points
            info = pick_points(args.input, args.output, args.template)
            print(f"\nPuntos elegidos: {info['picked']}  (pares: {info['pairs']})")
            print(f"Plantilla de referencias: {info['template']}")
            print("Completa real_distance_m en esa plantilla y luego usa 'holux-metric calibrate'.")
            return 0

        cfg = load_config(args.config)

        if args.command == "calibrate":
            from holux_metric.calibration import run_calibration
            res = run_calibration(args.input, args.references, args.planes, cfg, args.output)
            cal, meas = res["calibration"], res["measurements"]
            print(f"\nStatus: {cal['status']}   Escala s = {cal['scale']:.6f}   Salida: {args.output}")
            val = meas["summary_validation"]
            if val:
                print(f"Validación (n={val['n']}): MAE {_fmt_mm(val['MAE_m'])}, RMSE {_fmt_mm(val['RMSE_m'])}, "
                      f"máx {_fmt_mm(val['max_abs_error_m'])}, P95 {_fmt_mm(val['P95_abs_error_m'])}")
            for w in cal["warnings"]:
                print(f"  - {w}")
            return 0

        qa = run_o3d1(args.input, cfg, args.output, make_preview=not args.no_preview)
        print(f"\nStatus: {qa['status']}   Salida: {args.output}")
        for w in qa.get("warnings", []):
            print(f"  - {w}")
        return 1 if qa["status"] == "FAIL" else 0
    except (InputError, ConfigError, CalibrationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
