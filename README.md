# holux-metric — HO-LUX Spatial · capa de geometría métrica

Pipeline reproducible para transformar una nube de puntos de reconstrucción visual en
geometría limpia, medible y (más adelante) calibrada al sistema de coordenadas del Site.

**Estado:** O3D-1 (inspección + limpieza + normals + planos + QA) implementado.
Registration, Site Calibration y Metric Quality son esqueletos (`NotImplementedError`).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Open3D solo publica wheels para ciertas versiones de Python. Si `pip install` falla con
"no matching distribution", revisa tu versión de Python (`python --version`) y usa una compatible.

## Uso

```bash
# 1) Inspeccionar (imprime JSON; -o lo guarda)
holux-metric inspect input/sparse_pc.ply

# 2) O3D-1 completo
holux-metric clean input/sparse_pc.ply \
  --config config/o3d-site001.yaml \
  --output output/o3d-1/
```

Opciones: `--no-preview` (sin imágenes). Código de salida: `0` PASS/WARN, `1` FAIL, `2` error de uso.

## Pipeline O3D-1

```
LOAD → INSPECT → (descartar no finitos) → DOWNSAMPLE → OUTLIER FILTER → NORMALS → PLANES → QA
```

| Etapa | Módulo | Función de Open3D que usa | Salida |
|---|---|---|---|
| Carga/guardado | `io.py` | `read_point_cloud`, `write_point_cloud` | — |
| Inspección | `inspect.py` | `np.asarray(pcd.points)` | `inspect.json` |
| Limpieza | `clean.py` | `voxel_down_sample`, `remove_statistical_outlier`, `remove_radius_outlier` | `cleaning.json` |
| Normals | `normals.py` | `estimate_normals`, `orient_normals_consistent_tangent_plane` | (dentro del PLY) |
| Planos | `planes.py` | `segment_plane` (RANSAC, iterativo) | `planes.json` |
| QA | `qa.py` | — | `qa.json` |
| Previews | `preview.py` | — | `preview/*.png` |
| Orquestación | `pipeline.py` | — | `visual_reference_clean.ply` |

## Salida

```
output/o3d-1/
├── visual_reference_clean.ply
├── inspect.json
├── cleaning.json
├── planes.json
├── qa.json
└── preview/  input.png  cleaned.png  outliers.png  normals.png  planes.png
```

`qa.json` incluye `status` (PASS / WARN / FAIL), conteos antes/después, bounding boxes,
planos detectados, `warnings`, versiones (algoritmo, Open3D, Python) y los SHA-256 del input
y de la configuración, para poder reproducir cada corrida.

## Parámetros: lo más importante

Una nube de reconstrucción visual **no está en metros** (escala arbitraria). Por eso
`length_mode: relative` expresa voxel, radios y umbral de planos como **fracción de la
diagonal del bounding box**. Cuando la nube ya sea métrica, cambia a `absolute`.

Cómo afinar (itera sobre `config/o3d-site001.yaml` y mira `preview/` y `cleaning.json`):

1. `downsample.voxel_size`: sube si hay demasiados puntos o ejecución lenta; baja si pierdes estructura.
2. `outlier.statistical.std_ratio` (más bajo = más agresivo) y `outlier.radius.*`.
   `cleaning.json → outlier_comparison` compara ambos métodos; elige con `outlier.selected`.
3. `normals.radius`: debe cubrir unos 10–30 vecinos; mira `normals.png` (paredes y piso deberían tener color uniforme).
4. `planes.distance_threshold` y `planes.min_points`: si sale 0 planos o planos "gruesos", ajusta aquí.

## Mapa para estudiar los tutoriales de Open3D

- *Point cloud* → `io.py`, `inspect.py`
- *Voxel downsampling / Outlier removal* → `clean.py`
- *Vertex normal estimation* → `normals.py`
- *Plane segmentation* → `planes.py`
- *Global registration (FPFH + RANSAC) e ICP* → `registration.py` (siguiente etapa)

Para mirar la nube de forma interactiva:

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("output/o3d-1/visual_reference_clean.ply")
o3d.visualization.draw_geometries([pcd])
```

## Tests

```bash
pytest
```

Usan una nube sintética (piso + pared + ruido), así que no necesitan datos del Site.

## Convenciones

- Los directorios `CO/` y `NE/` del paquete de entrada no se procesan en O3D-1; se conservan intactos.
- `input/`, `output/` y `*.ply` están en `.gitignore`: los datos del Site no se suben al repositorio.
- Siguiente etapa: Registration → `registration.json`.
