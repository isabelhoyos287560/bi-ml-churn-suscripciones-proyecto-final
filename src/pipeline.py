"""Pipeline completo y reproducible: datos crudos → tablas procesadas → puntuación de riesgo → tablero (CSV para Power BI).

Uso:  python -m src.pipeline            (todo)
      python -m src.pipeline --sin-modelo   (solo ETL; omite puntuación y monitoreo, que tardan unos minutos)
"""
import sys
import time
from pathlib import Path

import pandas as pd

from . import etl, scoring, tablero

RAIZ = Path(__file__).resolve().parents[1]
RAW, PROC = RAIZ / "data" / "raw", RAIZ / "data" / "processed"


def paso(nombre, t0):
    print(f"[{time.time() - t0:6.0f} s] {nombre}")


def main(con_modelo=True):
    t0 = time.time()
    kpi, inf_kpi = etl.limpiar_kpi(etl.leer_kpi(RAW / "dataset suscriptores.csv"))
    ordenes, inf_ord = etl.limpiar_ordenes(etl.leer_ordenes(RAW / "dataset suscriptores original.csv"))
    ciclos = etl.construir_ciclos(ordenes)
    tablas = etl.construir_estrella(kpi, ciclos)
    val = etl.validar(inf_kpi, inf_ord, tablas)
    assert val["cumple"].all(), val[~val["cumple"]]
    etl.guardar(tablas, PROC)
    paso(f"ETL y validaciones ({int(val['cumple'].sum())}/{len(val)} controles)", t0)

    if con_modelo:
        snap = tablas["fact_snapshot_mensual"].copy()
        snap["fecha"] = pd.to_datetime(snap["fecha"])
        scoring.puntuar_abiertos(snap, ciclos).to_csv(PROC / "fact_riesgo_ciclo.csv", index=False)
        paso("Puntuación de ciclos abiertos", t0)
        tablero.cohortes(snap).to_csv(PROC / "fact_cohorte.csv", index=False)
        tablero.monitoreo_modelo(snap, ciclos).to_csv(PROC / "fact_monitoreo_modelo.csv", index=False)
        paso("Cohortes y monitoreo del modelo", t0)

    paso("Listo: tablas en data/processed/", t0)


if __name__ == "__main__":
    main(con_modelo="--sin-modelo" not in sys.argv)
