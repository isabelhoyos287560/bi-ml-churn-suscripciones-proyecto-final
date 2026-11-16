"""ETL del proyecto: lectura, limpieza, validaciones y modelo en estrella.

Dos fuentes unidas por `usuario_id` (hash del correo con sal):
- KPI: foto mensual de suscriptores activos (una fila por usuario y mes). Un usuario que
  deja de estar activo deja de aparecer; si renueva o se reactiva, vuelve a aparecer.
- Órdenes de suscripción: un ciclo por orden, con inicio, fin, producto, tipo de entrada
  y fecha de pago. Es la fuente de verdad para saber si un ciclo se renovó.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CORTE = pd.Timestamp("2026-10-31")      # último mes completo (noviembre 2026 es parcial)
HORIZONTE_DIAS = 90                     # ventana para decidir si el ciclo renovó
COLS_FECHA = ["fecha", "fecha_ultima_renovacion", "fecha_vencimiento_membresia"]
FECHA_MIN, FECHA_MAX_VENC = pd.Timestamp("2023-01-01"), pd.Timestamp("2030-12-31")
ID_VALIDO = r"[0-9a-f]{16}"             # hash del correo con sal, 16 caracteres hexadecimales
PRODUCTOS_PRINCIPALES = {"91113", "585066", "237481", "1091442", "237480"}   # suscripciones principales
ESTADOS_CANCELADOS = {"wc-cancelled", "wc-cancelled-pending", "wc-cancelled-request"}
TOPE_BUSQUEDA_DIAS = 455                # hasta cuántos días después del fin se busca la renovación


def leer_kpi(ruta):
    df = pd.read_csv(ruta, sep=";", encoding="cp1252", low_memory=False)
    for c in COLS_FECHA:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    return df


def limpiar_kpi(df):
    """Devuelve (df_limpio, informe). El informe permite cuadrar filas de entrada y salida."""
    inf = {"filas_entrada": len(df)}
    df = df.copy()

    # 1. Identificadores dañados (Excel los leyó como notación científica)
    ok = df["usuario_id"].astype(str).str.fullmatch(ID_VALIDO)
    inf["ids_danados_distintos"] = int(df.loc[~ok, "usuario_id"].nunique())
    inf["filas_id_danado"] = int((~ok).sum())
    df = df[ok]

    # 2. Duplicados exactos
    inf["filas_duplicadas"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. Categoría sin el prefijo "Suscripción "
    df["categoria"] = df["categoriamembresia"].str.replace("Suscripción ", "", regex=False).str.strip()
    df = df.drop(columns="categoriamembresia")

    # 4. Fechas de vencimiento imposibles o ausentes: se marcan, no se borran
    v = df["fecha_vencimiento_membresia"]
    imposible = v.notna() & ((v < FECHA_MIN) | (v > FECHA_MAX_VENC))
    inf["vencimientos_imposibles"] = int(imposible.sum())
    df.loc[imposible, "fecha_vencimiento_membresia"] = pd.NaT
    inf["filas_sin_vencimiento"] = int(df["fecha_vencimiento_membresia"].isna().sum())
    df["dias_hasta_vencimiento"] = (df["fecha_vencimiento_membresia"] - df["fecha"]).dt.days

    # 5. Mes parcial
    df["snapshot_parcial"] = df["fecha"] > CORTE

    # 6. Contadores no numéricos o negativos
    num = df.select_dtypes("number").columns.drop("dias_hasta_vencimiento", errors="ignore")
    inf["valores_negativos"] = int((df[num] < 0).sum().sum())

    inf["filas_salida"] = len(df)
    return df.reset_index(drop=True), inf


# ---------------------------------------------------------------- órdenes de suscripción
def leer_ordenes(ruta):
    o = pd.read_csv(ruta, dtype=str)
    for c in ["ciclo_inicio", "ciclo_fin", "fecha_pago"]:
        o[c] = pd.to_datetime(o[c], errors="coerce")
    return o


def limpiar_ordenes(o):
    """Aplica las reglas de negocio en orden y cuenta cuántas filas descarta cada una."""
    inf = {"filas_entrada": len(o)}
    o = o.copy()

    def paso(nombre, mantener):
        nonlocal o
        inf[nombre] = int((~mantener).sum())
        o = o[mantener]

    paso("descartadas_id_invalido", o["usuario_id"].astype(str).str.fullmatch(ID_VALIDO))
    paso("descartadas_orden_no_completada", o["ordenestado"] == "wc-completed")
    paso("descartadas_suscripcion_cancelada", ~o["suscripcionestado"].isin(ESTADOS_CANCELADOS))
    paso("descartadas_tipo_error", o["tipo_entrada"].fillna("") != "ERROR")
    ok = o["ciclo_inicio"].notna() & o["ciclo_fin"].notna() & (o["ciclo_fin"] >= o["ciclo_inicio"]) & (o["ciclo_fin"] <= FECHA_MAX_VENC)
    paso("descartadas_fechas_invalidas", ok)
    o = o.sort_values(["fecha_pago"], na_position="last")
    dup = o.duplicated(["usuario_id", "ordenid"]) | o.duplicated(["usuario_id", "ciclo_inicio", "ciclo_fin"])
    paso("descartadas_duplicadas", ~dup)

    o["tipo_entrada"] = o["tipo_entrada"].fillna("SIN_DATO")
    o["grupo_producto"] = np.where(o["productoid"].isin(PRODUCTOS_PRINCIPALES), "Principal", "Corto u otro")
    o["es_regalo"] = (o["tipo_entrada"] == "REGALO").astype(int)
    o["duracion_dias"] = (o["ciclo_fin"] - o["ciclo_inicio"]).dt.days
    inf["filas_salida"] = len(o)
    return o.reset_index(drop=True), inf


def construir_ciclos(o):
    """Un ciclo por orden. Para los de productos principales marca si hubo renovación.

    Renovación = el primer ciclo principal posterior del mismo usuario que empieza entre el
    vencimiento y 455 días después. `churn90` = 1 si no hay renovación o si empieza más de
    90 días después del vencimiento. Solo se calcula en ciclos maduros (vencimiento + 90 días
    ya pasó antes del corte).
    """
    c = o.sort_values(["usuario_id", "ciclo_inicio", "ciclo_fin"]).reset_index(drop=True)
    c["ciclo_id"] = np.arange(1, len(c) + 1)
    c["numero_ciclo"] = c.groupby("usuario_id").cumcount() + 1

    p = c[c["grupo_producto"] == "Principal"]
    sig = p[["ciclo_id", "usuario_id", "ciclo_inicio", "ciclo_fin"]].merge(
        p[["usuario_id", "ciclo_inicio", "fecha_pago"]].rename(columns={"ciclo_inicio": "ini_sig", "fecha_pago": "pago_sig"}),
        on="usuario_id")
    sig = sig[(sig["ini_sig"] >= sig["ciclo_fin"])
              & (sig["ini_sig"] <= sig["ciclo_fin"] + pd.Timedelta(days=TOPE_BUSQUEDA_DIAS))
              & (sig["ini_sig"] > sig["ciclo_inicio"])]
    ren = sig.groupby("ciclo_id").agg(inicio_renovacion=("ini_sig", "min"), fecha_pago_renovacion=("pago_sig", "min"))
    c = c.merge(ren, on="ciclo_id", how="left")

    c["dias_hasta_renovacion"] = (c["inicio_renovacion"] - c["ciclo_fin"]).dt.days
    c["madura"] = (c["grupo_producto"] == "Principal") & (c["ciclo_fin"] + pd.Timedelta(days=HORIZONTE_DIAS) <= CORTE)
    renovo = c["dias_hasta_renovacion"].le(HORIZONTE_DIAS)
    c["churn90"] = np.where(c["madura"], (~renovo).astype(int), np.nan)
    return c


def movimiento_mensual(kpi):
    """Altas, reactivaciones y bajas por mes a partir de las fotos (un usuario sale cuando no aparece al mes siguiente)."""
    m = kpi[["usuario_id", "fecha"]].copy()
    m["m"] = m["fecha"].dt.year * 12 + m["fecha"].dt.month
    m = m.sort_values(["usuario_id", "m"])
    prev, nxt = m.groupby("usuario_id")["m"].shift(1), m.groupby("usuario_id")["m"].shift(-1)
    m["tipo"] = np.where(prev.isna(), "nuevos", np.where(m["m"] - prev > 1, "reactivados", "continuan"))
    m["sale"] = nxt.isna() | (nxt - m["m"] > 1)
    t = m.groupby(["fecha", "tipo"]).size().unstack(fill_value=0)
    t["bajas"] = m[m["sale"]].groupby("fecha").size()
    t = t.fillna(0).astype(int).reset_index()
    t["activos"] = t[["continuan", "nuevos", "reactivados"]].sum(axis=1)
    primero, ultimos = t["fecha"].min(), sorted(t["fecha"])[-2:]
    # el primer mes (todos aparecen como nuevos) y los 2 últimos (aún no se sabe si salieron) no son comparables
    t["comparable"] = ~t["fecha"].isin([primero] + ultimos)
    return t


def construir_estrella(kpi, ciclos):
    dim_cat = pd.DataFrame({"categoria": ["Básica", "Oro", "Platino"], "categoria_id": [1, 2, 3]})

    meses = pd.date_range(kpi["fecha"].min(), kpi["fecha"].max(), freq="ME")
    dim_fecha = pd.DataFrame({"fecha": meses})
    dim_fecha["anio"] = dim_fecha["fecha"].dt.year
    dim_fecha["mes"] = dim_fecha["fecha"].dt.month
    dim_fecha["trimestre"] = dim_fecha["fecha"].dt.quarter
    dim_fecha["anio_mes"] = dim_fecha["fecha"].dt.strftime("%Y-%m")
    dim_fecha["snapshot_parcial"] = dim_fecha["fecha"] > CORTE

    dim_prod = (ciclos.groupby(["productoid", "grupo_producto"])
                .agg(ciclos=("ciclo_id", "size"), duracion_mediana_dias=("duracion_dias", "median"))
                .reset_index())

    f = kpi.merge(dim_cat, on="categoria", how="left").drop(columns="categoria")

    idx = kpi["fecha"].dt.year * 12 + kpi["fecha"].dt.month
    tmp = pd.DataFrame({"usuario_id": kpi["usuario_id"], "m": idx}).sort_values(["usuario_id", "m"])
    tmp["salto"] = tmp.groupby("usuario_id")["m"].diff().gt(1)
    k = tmp.groupby("usuario_id").agg(meses_activos=("m", "size"), reactivaciones=("salto", "sum")).reset_index()
    k = k.merge(kpi.groupby("usuario_id")["fecha"].agg(primer_mes="min", ultimo_mes="max").reset_index(), on="usuario_id")
    o = ciclos.groupby("usuario_id").agg(n_ciclos=("ciclo_id", "size"), primera_orden=("ciclo_inicio", "min")).reset_index()
    dim_usr = k.merge(o, on="usuario_id", how="outer")
    dim_usr["en_kpi"] = dim_usr["meses_activos"].notna()
    dim_usr["en_ordenes"] = dim_usr["n_ciclos"].notna()
    dim_usr["activo_ultimo_mes"] = dim_usr["ultimo_mes"] >= CORTE.replace(day=1)

    return {"dim_usuario": dim_usr, "dim_categoria": dim_cat, "dim_fecha": dim_fecha, "dim_producto": dim_prod,
            "fact_ciclo": ciclos, "fact_snapshot_mensual": f,
            "fact_movimiento_mensual": movimiento_mensual(kpi)}


def validar(inf_kpi, inf_ord, t):
    """Lista de (control, cumple, detalle)."""
    r = []
    s, c = t["fact_snapshot_mensual"], t["fact_ciclo"]
    du = t["dim_usuario"]
    r.append(("KPI · llave (usuario_id, fecha) única", not s.duplicated(["usuario_id", "fecha"]).any(), f"{len(s):,} filas"))
    r.append(("KPI · usuario_id con formato válido", s["usuario_id"].str.fullmatch(ID_VALIDO).all(), "16 hex"))
    r.append(("KPI · sin nulos en llaves", s[["usuario_id", "fecha", "categoria_id"]].notna().all().all(), ""))
    r.append(("KPI · fechas dentro de ene-2024 a nov-2026", s["fecha"].between("2024-01-01", "2026-11-30").all(), ""))
    r.append(("KPI · cuadre de filas", inf_kpi["filas_entrada"] == inf_kpi["filas_salida"] + inf_kpi["filas_id_danado"] + inf_kpi["filas_duplicadas"],
              f"{inf_kpi['filas_entrada']:,} = {inf_kpi['filas_salida']:,} + {inf_kpi['filas_id_danado']:,} + {inf_kpi['filas_duplicadas']:,}"))
    desc = sum(v for k, v in inf_ord.items() if k.startswith("descartadas_"))
    r.append(("Órdenes · cuadre de filas", inf_ord["filas_entrada"] == inf_ord["filas_salida"] + desc,
              f"{inf_ord['filas_entrada']:,} = {inf_ord['filas_salida']:,} + {desc:,}"))
    r.append(("Órdenes · ciclo_id único", c["ciclo_id"].is_unique, f"{len(c):,} ciclos"))
    r.append(("Órdenes · (usuario, orden) único", not c.duplicated(["usuario_id", "ordenid"]).any(), ""))
    r.append(("Órdenes · fin del ciclo no anterior al inicio", (c["ciclo_fin"] >= c["ciclo_inicio"]).all(), ""))
    r.append(("churn90 solo en ciclos maduros de productos principales", c.loc[~c["madura"], "churn90"].isna().all(), f"{int(c['madura'].sum()):,} maduros"))
    r.append(("Integridad: usuarios de las dos fuentes en dim_usuario",
              s["usuario_id"].isin(du["usuario_id"]).all() and c["usuario_id"].isin(du["usuario_id"]).all(), f"{len(du):,} usuarios"))
    r.append(("Integridad: categorías y meses en sus dimensiones",
              s["categoria_id"].isin(t["dim_categoria"]["categoria_id"]).all() and s["fecha"].isin(t["dim_fecha"]["fecha"]).all(), ""))
    r.append(("Integridad: productos de los ciclos en dim_producto", c["productoid"].isin(t["dim_producto"]["productoid"]).all(), ""))
    cob = float(du.loc[du["en_kpi"], "en_ordenes"].mean())
    r.append(("Cobertura: usuarios del KPI con órdenes (≥ 99 %)", cob >= 0.99, f"{cob:.1%}"))
    r.append(("Sin valores negativos en contadores del KPI", inf_kpi["valores_negativos"] == 0, f"{inf_kpi['valores_negativos']:,} negativos"))
    return pd.DataFrame(r, columns=["control", "cumple", "detalle"])


def guardar(tablas, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, t in tablas.items():
        t.to_csv(carpeta / f"{nombre}.csv", index=False, encoding="utf-8", date_format="%Y-%m-%d")
