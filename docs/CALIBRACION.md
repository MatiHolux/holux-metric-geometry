# Calibración al Site (P_site = s · R · P_visual + t)

Convierte la nube (coordenadas visuales, escala arbitraria) a **metros** con un sistema de
coordenadas del Site (Z arriba, origen definido).

## Flujo

1. **Elegir puntos** en la nube (idealmente `visual_reference_clean.ply`):

   ```powershell
   holux-metric pick output\o3d-1\visual_reference_clean.ply
   ```

   Shift + clic izquierdo para marcar, **Q** para terminar. Marca los puntos **de a pares**
   (A, B, A, B, ...): cada par es una distancia que también mediste en el lugar (puerta, rack, pared a pared).
   Genera `output\calibration\references.template.yaml`.

2. **Completar la plantilla** (cópiala a `config\references-site001.yaml`):
   - `real_distance_m`: la medida física en metros.
   - `role: calibrate` para las que ajustan la escala; `role: validate` para las que reservas
     para comprobar el error (no participan en el ajuste). Usa **al menos 2 de calibración y 3-5 de validación**.
   - Opcional, `frame`: `floor_plane_id` (p. ej. `plane_001`, es el plano del piso en `planes.json`;
     el color de cada plano en `planes.png` sigue el orden azul, naranjo, verde, rojo, morado, café),
     `x_axis` (dos puntos a lo largo de una pared) y `origin_visual`.

3. **Calibrar**:

   ```powershell
   holux-metric calibrate output\o3d-1\visual_reference_clean.ply `
     --references config\references-site001.yaml `
     --planes output\o3d-1\planes.json `
     --output output\calibration
   ```

## Salidas (`output\calibration\`)

| Archivo | Contenido |
|---|---|
| `site_reference_metric.ply` | La nube transformada (metros, Z arriba) |
| `site_calibration.json` | `s`, `R`, `t`, matriz 4x4, ejes, diagnóstico de la escala, hashes de entradas, warnings |
| `measurements.json` | Por referencia: medida real, medida del modelo y error; MAE, RMSE, error máximo y P95 |

## Cómo leer los resultados

- **Los estadísticos que importan son los de `summary_validation`** (medidas no usadas para calibrar).
  Los de `summary_calibration` son optimistas: el ajuste los minimiza a propósito.
- `ratio_cv_percent` alto significa que las referencias de calibración se contradicen entre sí
  (puntos mal elegidos, medidas mal tomadas o una reconstrucción deformada).
- Con pocas validaciones, P95 y RMSE no son representativos (el programa lo advierte).
- Una escala uniforme corrige el tamaño, pero **no** deformaciones de la reconstrucción:
  si el error de validación cambia según la zona, el modelo está deformado.
- El umbral `calibration.max_validation_error_m` (config) define cuándo avisar; acuérdalo con tu jefe.

## Parámetros opcionales en `config\o3d-site001.yaml`

Si no los pones, se usan estos valores:

```yaml
calibration:
  warn_scale_cv_percent: 2.0
  max_validation_error_m: 0.05
  min_validation_refs_for_p95: 5
```
