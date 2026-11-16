# Predicción de churn a 90 días en suscripciones: BI + analítica + ML

Proyecto final integrador (Análisis de datos, ITM, 2026-2). Predice qué ciclos de suscripción no se renovarán en los siguientes 90 días, con datos anonimizados de una empresa de suscripciones, y convierte el resultado en un tablero para decidir a quién contactar.

## Fases del proyecto → notebooks

| Fase | Notebook |
|---|---|
| 1. Recopilación y transformación (ETL) | `01_etl_limpieza.ipynb` (también `src/etl.py`) |
| 2. Análisis exploratorio (EDA) | `02_eda.ipynb` |
| 3. Inteligencia de negocios (modelo de datos del tablero) | `src/tablero.py` (tablas en estrella) |
| 4. Modelado supervisado y métricas | `03_modelado_supervisado.ipynb`, `06_validacion_y_ajuste.ipynb` |
| Segmentación y causas | `04_clustering.ipynb`, `05_techo_y_causas.ipynb` |
| 5. Conclusiones y recomendaciones | cierre de cada notebook |

## Cómo reproducirlo

```bash
pip install -r requirements.txt
# Colocar los datos crudos exportados en data/raw/ (no se suben al repositorio por ser datos de la empresa):
#    "dataset suscriptores.csv" y "dataset suscriptores original.csv"
python -m src.pipeline                 # ETL + puntuación de riesgo + tablas del tablero -> data/processed/
python -m src.pipeline --sin-modelo    # solo ETL (rápido)
jupyter lab                            # ejecutar los notebooks en orden 01 → 06
```

`python -m src.pipeline` es el ETL en un solo comando: lee los CSV exportados desde PostgreSQL, transforma y valida en Python, y escribe los CSV procesados (modelo en estrella). No se monta ninguna base de datos adicional.

## Estructura

```
data/raw/         datos crudos exportados (no versionados)
data/processed/   tablas en estrella (no versionadas)
src/              etl.py, features.py, scoring.py, tablero.py, pipeline.py
notebooks/        01 ETL · 02 EDA · 03 modelo · 04 clustering · 05 techo y causas · 06 validación
```

## Modelo de datos

Esquema en estrella en `data/processed/`:
- Hechos: `fact_ciclo`, `fact_snapshot_mensual`, `fact_movimiento_mensual`.
- Dimensiones: `dim_usuario`, `dim_producto`, `dim_categoria`.
- Resultados del modelo: `fact_riesgo_ciclo` (nivel, riesgo, acción sugerida y prioridad por ciclo), `fact_cohorte`, `fact_monitoreo_modelo` (AUC y separación por mes).

## Resultados clave

- Churn a 90 días: 49,6 % sobre 74.421 ciclos maduros.
- AUC mensual en el seguimiento (mar–jul 2026): entre 0,62 y 0,75, promedio ≈ 0,68.
- Los niveles de riesgo separan: churn real de 85 % (Alto), 65 % (Medio) y 52 % (Bajo).
- El mayor riesgo está en las altas nuevas del plan Básica (86 % de churn).

## Nota sobre los datos

Los datos están anonimizados (identificadores en hash) y no se publican. Para ejecutar el proyecto se necesitan los dos CSV crudos descritos arriba.
