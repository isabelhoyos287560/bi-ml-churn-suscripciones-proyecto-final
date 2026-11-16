"""Puntuación de riesgo de los ciclos abiertos (uso operativo).

Para cada ciclo que vence en los próximos ~4 meses y aún no tiene la renovación pagada:
1. se elige el modelo del horizonte más cercano (4, 3, 2 o 1 mes antes del vencimiento),
2. se calcula el riesgo con datos hasta el corte de información,
3. se recalibra el nivel (Platt, que evita probabilidades de 0 o 1) con los resultados más recientes (los 3 últimos meses con resultado conocido),
4. se asigna un nivel (Alto / Medio / Bajo) según el lugar en el ranking del lote,
5. se informa el churn esperado de cada nivel tomado de la prueba temporal (lo que realmente pasó), no de la probabilidad.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import features

HOY = pd.Timestamp("2026-10-31")


def _modelo():
    pre = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), features.CAT)],
                            remainder="passthrough")
    return make_pipeline(pre, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5,
                                                             min_samples_leaf=40, l2_regularization=1.0, random_state=0))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _x(df, cols):
    x = df[cols].copy()
    for k in features.CAT:
        x[k] = x[k].astype(str)
    return x


def churn_por_nivel_backtest(snapshot, ciclos):
    """Churn real por nivel (Alto/Medio/Bajo) en la prueba temporal (vencimientos may-jul 2026, modelo entrenado hasta ene-2026).

    Es la cifra honesta para comunicar: el puntaje sirve para ordenar, pero el churn esperado de cada nivel
    se toma de lo que realmente ocurrió, no de la probabilidad del modelo, que en la zona baja es optimista."""
    filas = []
    for meses, dias in features.HORIZONTES.items():
        b = features.base_horizonte(snapshot, ciclos, dias)
        cols = features.columnas_modelo(b)
        tr = b[b["ciclo_fin"] <= "2026-01-31"]
        te = b[(b["ciclo_fin"] >= "2026-05-01") & (b["ciclo_fin"] <= "2026-07-31")].copy()
        te["p"] = _modelo().fit(_x(tr, cols), tr["churn90"]).predict_proba(_x(te, cols))[:, 1]
        pos = te["p"].rank(pct=True, method="first")
        te["nivel"] = np.where(pos > 0.75, "Alto", np.where(pos > 0.25, "Medio", "Bajo"))
        g = te.groupby("nivel")["churn90"].agg(churn_esperado="mean", n_prueba="size").reset_index()
        g.insert(0, "horizonte_meses", meses)
        filas.append(g)
    return pd.concat(filas, ignore_index=True)


def puntuar_abiertos(snapshot, ciclos, hoy=HOY):
    salidas = []
    for meses, dias in features.HORIZONTES.items():
        hist = features.base_horizonte(snapshot, ciclos, dias)
        cols = features.columnas_modelo(hist)
        ultimo = hist["ciclo_fin"].max()
        # recalibración: modelo entrenado antes de los últimos 3 meses con resultado, isotónica sobre esos 3 meses
        corte_cal = ultimo - pd.DateOffset(months=3)
        a, b = hist[hist["ciclo_fin"] <= corte_cal], hist[hist["ciclo_fin"] > corte_cal]
        m0 = _modelo().fit(_x(a, cols), a["churn90"])
        platt = LogisticRegression(C=100).fit(_logit(m0.predict_proba(_x(b, cols))[:, 1]), b["churn90"])   # recalibración suave (Platt)
        m = _modelo().fit(_x(hist, cols), hist["churn90"])
        ab = features.base_horizonte(snapshot, ciclos, dias, abiertos=True, hoy=hoy)
        dias_restantes = (ab["ciclo_fin"] - hoy).dt.days
        # cada ciclo usa el horizonte que mejor corresponde a los días que le quedan
        lo, hi = {4: (105, 9999), 3: (75, 105), 2: (45, 75), 1: (-9999, 45)}[meses]
        ab = ab[(dias_restantes > lo) & (dias_restantes <= hi)].copy() if meses != 1 else ab[dias_restantes <= hi].copy()
        if ab.empty:
            continue
        ab["horizonte_meses"] = meses
        ab["riesgo_bruto"] = m.predict_proba(_x(ab, cols))[:, 1]
        ab["riesgo"] = platt.predict_proba(_logit(ab["riesgo_bruto"].values))[:, 1]
        salidas.append(ab)
    r = pd.concat(salidas, ignore_index=True)
    r["posicion"] = r["riesgo_bruto"].rank(pct=True, method="first")
    r["nivel"] = np.where(r["posicion"] > 0.75, "Alto", np.where(r["posicion"] > 0.25, "Medio", "Bajo"))
    r["dias_para_vencer"] = (r["ciclo_fin"] - hoy).dt.days
    bt = churn_por_nivel_backtest(snapshot, ciclos)
    r = r.merge(bt[["horizonte_meses", "nivel", "churn_esperado"]], on=["horizonte_meses", "nivel"], how="left")
    cols_out = ["ciclo_id", "usuario_id", "ciclo_inicio", "ciclo_fin", "dias_para_vencer", "horizonte_meses", "productoid",
                "tipo_entrada", "categoria_id", "ciclos_previos", "antiguedad_meses", "nav_media", "nav_ult3", "dias_inactividad_ult", "tendencia_nav", "riesgo", "nivel", "churn_esperado"]
    r = r[cols_out].copy()
    r = _acciones(r)
    return r.sort_values(["prioridad_orden", "riesgo"], ascending=[True, False]).reset_index(drop=True)


PRODUCTO_ALTO_CHURN = {"237480"}      # producto con ~81 % de churn histórico (ver notebook 05)


def _acciones(r):
    """Reglas simples y explicables que convierten el nivel de riesgo en una acción sugerida.
    Son reglas de negocio basadas en los hallazgos del EDA y del notebook 05, no salida del modelo."""
    sin_uso = r["nav_ult3"].fillna(0) == 0
    r["perfil_uso"] = np.select(
        [r["nav_ult3"].isna() | (r["dias_inactividad_ult"].fillna(999) > 180) & sin_uso, sin_uso,
         r["nav_ult3"] >= 2.5],
        ["Sin uso", "Uso que se apagó", "Muy activo"], default="Uso moderado")
    nuevo = r["tipo_entrada"] == "NUEVA"
    producto_riesgo = r["productoid"].isin(PRODUCTO_ALTO_CHURN)
    r["motivo_principal"] = np.select(
        [producto_riesgo, nuevo & (r["perfil_uso"] == "Sin uso"), nuevo, (r["tipo_entrada"] == "REACTIVACION"),
         r["perfil_uso"].isin(["Sin uso", "Uso que se apagó"])],
        ["Producto con churn histórico muy alto", "Alta nueva sin uso", "Primera renovación", "Reactivación (poco predecible)",
         "Cliente con historia que dejó de usar"], default="Renovación habitual")
    r["accion_sugerida"] = np.select(
        [producto_riesgo, nuevo & (r["perfil_uso"] == "Sin uso"), nuevo, (r["tipo_entrada"] == "REACTIVACION"),
         r["perfil_uso"].isin(["Sin uso", "Uso que se apagó"])],
        ["Revisar oferta y proponer cambio de plan", "Activación: primeros pasos y contenido de bienvenida",
         "Acompañar la primera renovación; ofrecer plan largo", "Comunicación estándar de renovación",
         "Reenganche con contenido de su perfil"], default="Renovación estándar automática")
    urgente = (r["dias_para_vencer"] <= 30) & (r["nivel"] == "Alto")
    r["prioridad"] = np.select([urgente, r["nivel"] == "Alto", r["nivel"] == "Medio"], ["1 · Urgente", "2 · Alta", "3 · Media"], default="4 · Baja")
    r["prioridad_orden"] = r["prioridad"].str[0].astype(int)
    return r
