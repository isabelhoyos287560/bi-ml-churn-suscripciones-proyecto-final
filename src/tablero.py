"""Tablas adicionales para el tablero: cohortes de retención y monitoreo mensual del modelo."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import features, scoring


def cohortes(snapshot):
    """Retención por cohorte de alta: de los usuarios que aparecen por primera vez en un mes, cuántos siguen activos k meses después."""
    s = snapshot[["usuario_id", "fecha"]].copy()
    s["fecha"] = pd.to_datetime(s["fecha"])
    s["m"] = s["fecha"].dt.year * 12 + s["fecha"].dt.month - 1
    s["cohorte_m"] = s.groupby("usuario_id")["m"].transform("min")
    s["k"] = s["m"] - s["cohorte_m"]
    g = s.groupby(["cohorte_m", "k"])["usuario_id"].nunique().rename("activos").reset_index()
    tam = g[g["k"] == 0].set_index("cohorte_m")["activos"].rename("tamaño")
    g = g.join(tam, on="cohorte_m")
    g["retencion"] = g["activos"] / g["tamaño"]
    g["cohorte"] = pd.to_datetime(((g["cohorte_m"] // 12).astype(str) + "-" + (g["cohorte_m"] % 12 + 1).astype(str).str.zfill(2) + "-01"))
    ultimo = s["m"].max()
    g = g[g["cohorte_m"] + g["k"] <= ultimo - 1]                # sin el mes parcial
    g = g[g["cohorte_m"] > s["m"].min()]                         # la primera foto no es una cohorte real (todos aparecen como nuevos)
    return g[["cohorte", "k", "tamaño", "activos", "retencion"]].rename(columns={"k": "meses_desde_alta"})


def monitoreo_modelo(snapshot, ciclos, meses_atras=6, dias=60):
    """Reentrena cada mes con lo que se sabía entonces y mide cómo le fue en los vencimientos de ese mes.
    Sirve para vigilar si el modelo mejora o empeora."""
    b = features.base_horizonte(snapshot, ciclos, dias)
    cols = features.columnas_modelo(b)
    b["mes"] = b["ciclo_fin"].dt.to_period("M").dt.to_timestamp()
    filas = []
    for mes in sorted(b["mes"].unique())[-meses_atras:]:
        mes = pd.Timestamp(mes)
        tr = b[b["ciclo_fin"] + pd.Timedelta(days=90) <= mes]          # solo ciclos con resultado conocido antes de ese mes
        te = b[b["mes"] == mes]
        if len(te) < 200 or te["churn90"].nunique() < 2:
            continue
        p = scoring._modelo().fit(scoring._x(tr, cols), tr["churn90"]).predict_proba(scoring._x(te, cols))[:, 1]
        pos = pd.Series(p, index=te.index).rank(pct=True, method="first")
        y = te["churn90"]
        filas.append({"mes_vencimiento": mes, "ciclos": len(te), "churn_real": y.mean(), "auc": roc_auc_score(y, p),
                      "churn_nivel_alto": y[pos > 0.75].mean(), "churn_nivel_bajo": y[pos <= 0.25].mean(),
                      "diferencia_alto_bajo": y[pos > 0.75].mean() - y[pos <= 0.25].mean()})
    return pd.DataFrame(filas)
