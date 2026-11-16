"""Base analítica por ciclo y horizonte.

Cada fila es un ciclo de suscripción principal visto a `h` días de su vencimiento. Solo se usa
información anterior a esa fecha de corte (`corte_obs = ciclo_fin - h`):
- atributos del ciclo (tipo de entrada, producto, duración) e historial previo del usuario,
- comportamiento de las fotos mensuales con fecha <= corte_obs y dentro del ciclo.
Se excluyen los usuarios que ya habían pagado la renovación al corte (no hay nada que predecir).
"""
import numpy as np
import pandas as pd

HORIZONTES = {4: 120, 3: 90, 2: 60, 1: 30}
USO = ["total_dias_navegacion_mes", "total_apertura_boletin", "total_descarga_mformatos", "total_preguntas_ia",
       "total_peticiones_calculadoras", "total_peticiones_guias", "total_peticiones_bookstack",
       "total_peticiones_uamts", "total_peticiones_r_actualizacion", "total_solicitudes_soporte",
       "total_contenido_favorito", "total_respuestas_quices", "total_interacciones_learndash",
       "total_clics_consultorios", "total_contenido_compartido", "total_peticiones_lchequeo"]
CONTEXTO = ["cantidad_cupones_aplicados", "valor_descuentos_cupones", "total_compras", "total_compras_no_suscripcion",
            "total_tickets_fidelizacion", "total_aperturas_onboarding", "total_espacios_agente", "total_clientes_planner",
            "total_certificaciones_learndash", "dias_hasta_primer_ingreso"]


def _historial(ciclos):
    c = ciclos.sort_values(["usuario_id", "ciclo_inicio"]).copy()
    g = c.groupby("usuario_id")
    c["ciclos_previos"] = g.cumcount()
    c["pago_vs_inicio"] = (c["ciclo_inicio"] - c["fecha_pago"]).dt.days          # >0: pagó antes de empezar
    c["prev_pago_vs_inicio"] = g["pago_vs_inicio"].shift(1)
    c["prev_dias_hasta_renovacion"] = g["dias_hasta_renovacion"].shift(1)       # solo ciclos ya cerrados
    c["prev_tipo_entrada"] = g["tipo_entrada"].shift(1).fillna("NINGUNO")
    c["prev_duracion"] = g["duracion_dias"].shift(1)
    c["regalos_previos"] = g["es_regalo"].cumsum() - c["es_regalo"]
    c["dias_desde_primer_ciclo"] = (c["ciclo_inicio"] - g["ciclo_inicio"].transform("min")).dt.days
    return c


LARGO = ["fecha", "total_dias_navegacion_mes", "total_apertura_boletin", "cantidad_cupones_aplicados", "valor_descuentos_cupones",
         "total_compras", "dias_inactividad", "total_peticiones_r_actualizacion", "total_descarga_mformatos", "contacto_mes"]


def _historia_larga(snapshot, c):
    """Comportamiento de los 12 meses anteriores al corte, sin importar en qué ciclo ocurrió.
    Permite ver el uso reciente (6 meses) frente al previo y la continuidad de las fotos."""
    s = snapshot[["usuario_id"] + LARGO].copy()
    s["fecha"] = pd.to_datetime(s["fecha"])
    m = c[["ciclo_id", "usuario_id", "corte_obs"]].merge(s, on="usuario_id")
    m = m[(m["fecha"] <= m["corte_obs"]) & (m["fecha"] > m["corte_obs"] - pd.Timedelta(days=366))].copy()
    m["meses_atras"] = ((m["corte_obs"] - m["fecha"]).dt.days / 30.4).astype(int)
    g = m.groupby("ciclo_id")
    r6, p6, r3 = m[m["meses_atras"] < 6].groupby("ciclo_id"), m[m["meses_atras"] >= 6].groupby("ciclo_id"), m[m["meses_atras"] < 3].groupby("ciclo_id")
    f = pd.DataFrame(index=g.size().index)
    f["L_fotos_12m"] = g.size()
    f["L_fotos_6m"] = r6.size()
    f["L_nav_6m"] = r6["total_dias_navegacion_mes"].mean()
    f["L_nav_prev6m"] = p6["total_dias_navegacion_mes"].mean()
    f["L_nav_cambio"] = f["L_nav_6m"] - f["L_nav_prev6m"]
    f["L_nav_3m"] = r3["total_dias_navegacion_mes"].mean()
    f["L_boletin_3m"] = r3["total_apertura_boletin"].mean()
    f["L_cupones_12m"] = g["cantidad_cupones_aplicados"].sum()
    f["L_descuento_12m"] = g["valor_descuentos_cupones"].sum()
    f["L_compras_12m"] = g["total_compras"].sum()
    f["L_actualiz_6m"] = r6["total_peticiones_r_actualizacion"].mean()
    f["L_formatos_6m"] = r6["total_descarga_mformatos"].mean()
    f["L_inactividad_max"] = g["dias_inactividad"].max()
    f["L_contactos_12m"] = g["contacto_mes"].apply(lambda x: (x > 0).sum())
    return f.reset_index()


def base_horizonte(snapshot, ciclos, horizonte_dias, abiertos=False, hoy=None, excluir_pagados=True):
    c = _historial(ciclos)
    if abiertos:
        # ciclos que vencen en los próximos ~4 meses y cuyo resultado aún no se conoce (uso operativo)
        hoy = pd.Timestamp(hoy)
        c = c[(c["grupo_producto"] == "Principal") & (c["es_regalo"] == 0) & (~c["madura"])
              & (c["ciclo_fin"] >= hoy) & (c["ciclo_fin"] <= hoy + pd.Timedelta(days=130))].copy()
        c["corte_obs"] = (c["ciclo_fin"] - pd.Timedelta(days=horizonte_dias)).clip(upper=hoy)
    else:
        c = c[(c["grupo_producto"] == "Principal") & (c["es_regalo"] == 0) & c["madura"]].copy()
        c["corte_obs"] = c["ciclo_fin"] - pd.Timedelta(days=horizonte_dias)
    # ya había pagado la renovación al corte -> fuera
    c["ya_pago"] = c["fecha_pago_renovacion"].notna() & (c["fecha_pago_renovacion"] <= c["corte_obs"])
    if abiertos:
        c["ya_pago"] = c["fecha_pago_renovacion"].notna()
    c["pago_anticipado"] = c["ya_pago"].astype(int)       # solo se usa para mostrar el efecto de NO excluirlos
    if excluir_pagados:
        c = c[c["ya_pago"] == False]
    c = c[c["corte_obs"] > c["ciclo_inicio"]].copy()
    c["duracion_extra"] = (c["duracion_dias"] > 380).astype(int)   # plan de ~13 meses
    c["mes_vencimiento"] = c["ciclo_fin"].dt.month
    c["dias_observables"] = (c["corte_obs"] - c["ciclo_inicio"]).dt.days

    s = snapshot[["usuario_id", "fecha", "categoria_id", "antiguedad_suscriptor_meses", "numero_renovaciones_usuario",
                  "dias_inactividad", "contacto_mes", "objecion_producto_mes"] + USO + CONTEXTO].copy()
    s["fecha"] = pd.to_datetime(s["fecha"])
    m = c[["ciclo_id", "usuario_id", "ciclo_inicio", "corte_obs"]].merge(s, on="usuario_id")
    m = m[(m["fecha"] <= m["corte_obs"]) & (m["fecha"] >= m["ciclo_inicio"] - pd.Timedelta(days=31))]
    m = m.sort_values(["ciclo_id", "fecha"])
    g = m.groupby("ciclo_id")
    f = pd.DataFrame(index=g.size().index)
    f["meses_observados"] = g.size()
    f["categoria_id"] = g["categoria_id"].last()
    f["antiguedad_meses"] = g["antiguedad_suscriptor_meses"].last()
    f["dias_inactividad_ult"] = g["dias_inactividad"].last()
    f["nav_ultimo"] = g["total_dias_navegacion_mes"].last()
    f["nav_media"] = g["total_dias_navegacion_mes"].mean()
    f["nav_max"] = g["total_dias_navegacion_mes"].max()
    f["meses_con_uso"] = g["total_dias_navegacion_mes"].apply(lambda x: (x > 0).sum())
    ult3 = m.groupby("ciclo_id").tail(3).groupby("ciclo_id")["total_dias_navegacion_mes"].mean()
    f["nav_ult3"] = ult3
    f["tendencia_nav"] = f["nav_ult3"] - f["nav_media"]
    for col in USO:
        if col != "total_dias_navegacion_mes":
            f[col.replace("total_", "") + "_prom"] = g[col].mean()
    for col in CONTEXTO:
        f[col.replace("total_", "") + "_sum"] = g[col].sum()
    f["contactos_n"] = g["contacto_mes"].apply(lambda x: (x > 0).sum())      # el campo existe solo desde feb-2026
    f["objeciones_n"] = g["objecion_producto_mes"].apply(lambda x: (x > 0).sum())
    f["dias_primer_ingreso_prom"] = g["dias_hasta_primer_ingreso"].first()
    f["variedad_uso"] = (g[USO].sum() > 0).sum(axis=1)
    c = c.merge(f.reset_index(), on="ciclo_id", how="left")
    c = c.merge(_historia_larga(snapshot, c), on="ciclo_id", how="left")
    c["sin_foto_previa"] = c["meses_observados"].isna().astype(int)
    c["horizonte_dias"] = horizonte_dias
    return c


NUM_BASE = ["duracion_dias", "numero_ciclo", "ciclos_previos", "regalos_previos", "dias_desde_primer_ciclo",
            "duracion_extra", "mes_vencimiento", "dias_observables", "sin_foto_previa", "meses_observados",
            "categoria_id", "antiguedad_meses", "dias_inactividad_ult", "nav_ultimo", "nav_media", "nav_max",
            "meses_con_uso", "nav_ult3", "tendencia_nav", "variedad_uso", "pago_vs_inicio", "prev_pago_vs_inicio",
            "prev_dias_hasta_renovacion", "prev_duracion", "contactos_n", "objeciones_n"]
CAT = ["tipo_entrada", "productoid", "prev_tipo_entrada"]


# `contacto_mes` y `objecion_producto_mes` solo se registran desde feb-2026: el modelo no puede aprender de ellas con
# el histórico de entrenamiento, así que se usan solo para describir (EDA y análisis del contacto), no como predictoras.
SOLO_DESCRIPTIVAS = ["contactos_n", "objeciones_n", "L_contactos_12m"]


def columnas_modelo(df):
    extra = [c for c in df.columns if c.endswith("_prom") or c.endswith("_sum") or c.startswith("L_")]
    return [c for c in NUM_BASE + extra if c not in SOLO_DESCRIPTIVAS] + CAT
