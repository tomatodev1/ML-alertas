"""Dataset maestro Track A regional — Áncash, unidad = UGT × semana.

Reactiva Track A usando la BD interna de incidentes como fuente de label con
resolución SEMANAL (lo que Defensoría, mensual, no permitía). La unidad es la
UGT de ANTAMINA (Mina San Marcos, Huallanca, Valle Fortaleza, Huarmey), que es
la unidad operativa real del cliente y da un label no trivial.

Convención de semana (desde 2026-11-18): cada fila se identifica por
`semana_inicio` (lunes) y `semana_fin` (domingo de cierre). La semana que se
exporta al dashboard es `semana_fin`.

Anti-fuga: las features de la semana t usan SOLO datos con fecha <= semana_fin
(domingo); el label y_h mira la ventana futura (semana_fin, semana_fin + h].
Antes de esta fecha el label arrancaba en semana_inicio + 1 día, solapando 6
días con la ventana de `inc_prot_1w` (mismo evento como feature y como label).
"""
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.ancash_datos import JERARQUIA, _cargar_incidentes_raw, _norm

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
SALIDA = PROCESSED_DIR / "master_ancash_ugt.parquet"

UGTS = ["Mina San Marcos", "Huallanca", "Valle Fortaleza", "Huarmey"]
HORIZONTES = [14, 30, 60]
FECHA_INICIO = "2024-01-01"
# Fecha de corte de datos observados: se calcula en construir() como la
# última fecha con eventos en las fuentes de label (ver _fecha_corte_datos).
# Un label y_h solo es válido si semana_fin + h <= FECHA_CORTE.
FECHA_CORTE: pd.Timestamp | None = None

# Inicio de cobertura continua y fiable de dataIncidentes + dataFormulario.
# Antes de esta fecha el label viene de la Defensoría mensual (proxy menos sensible).
FECHA_INICIO_INCIDENTES = pd.Timestamp("2023-08-18")

# ── EXTENSIÓN DE VENTANA (bloqueada) ──────────────────────────────────────────
# Se descargaron 62 PDFs de la Defensoría (2018-2023, 86% cobertura) y la
# infraestructura está lista en _incidentes_defensoria_hist(). Con
# FECHA_INICIO="2018-01-01" el dataset pasa de 444 a 1,696 filas. Dos intentos
# de desbloqueo fallaron operativamente:
#   1. RF sin calibrar: probabilidades comprimidas ~63% para todas las UGTs
#   2. RF con CalibratedClassifierCV(isotonic, cv=TimeSeriesSplit(3)):
#      calibración se ancló a la tasa histórica (27.2%) en vez de la de
#      deployment (43.9%) → todo BAJO, 22-26%, sin spread útil
# LR tampoco mejora con la extensión: rep_antamina_neg_4w / rep_compromiso_4w
# son 0 en 2018-2023 y diluyen los coeficientes → PR-AUC baja 0.8004 → 0.4929.
# Solución correcta pendiente: calibrar isotónica SOLO sobre predicciones OOF
# del período 2024-2026 (distribución de deployment). Requiere cambios al
# framework de evaluación. Hasta entonces, FECHA_INICIO="2024-01-01" + LR.

ZONA_DEFENSORIA = "Áncash"  # las 4 UGTs comparten el contexto departamental


# ── Utilidades compartidas (antes en build_master.py) ────────────────────────
def _floor_to_monday(fechas: pd.Series) -> pd.Series:
    return fechas.dt.normalize() - pd.to_timedelta(fechas.dt.weekday, unit="D")


def _fin_semana(semana_inicio) -> np.ndarray:
    """Domingo de cierre de cada semana, como datetime64[ns]."""
    return np.asarray(semana_inicio, dtype="datetime64[ns]") + np.timedelta64(6, "D")


def _fecha_corte_datos() -> pd.Timestamp:
    """Último día observado en las fuentes de label: el MÍNIMO entre la última
    fecha de dataIncidentes y la de dataFormulario (conservador: un día solo
    cuenta como observado si ambas fuentes ya lo cubren)."""
    from src.scoring.ancash_datos import _cargar_formulario_protestas_raw
    inc = _cargar_incidentes_raw()
    max_inc = inc.loc[inc["fuente"] == "dataIncidentes", "fecha"].max()
    max_form = _cargar_formulario_protestas_raw()["fecha"].max()
    candidatos = [f for f in (max_inc, max_form) if pd.notna(f)]
    return min(candidatos).normalize()


def _racha_consecutiva(s: pd.Series) -> pd.Series:
    """Semanas consecutivas con al menos 1 evento (se reinicia en 0 al no haber)."""
    tiene = (s > 0).astype(int)
    grupos = (tiene != tiene.shift().fillna(0)).cumsum()
    return tiene.groupby(grupos).cumsum()


def _cargar_defensoria() -> pd.DataFrame:
    ruta = Path("data/raw/defensoria/defensoria_historico.csv")
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Features/labels de Defensoría serán NaN.")
        return pd.DataFrame(
            columns=["zona", "año", "mes_num", "escalamiento_zona", "escalamiento_zona_prev", "escalamiento_global"]
        )
    df = pd.read_csv(ruta)
    for col in ["escalamiento_zona", "escalamiento_zona_prev", "escalamiento_global"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Mes aún no publicado: la API devuelve 0 en el KPI global (nunca 0 en un
    # mes real). Se trata como faltante para que el ffill use el último
    # reporte efectivamente publicado, en vez de un falso "escalamiento 0".
    if "escalamiento_global" in df.columns:
        no_publicado = df["escalamiento_global"].fillna(0) == 0
        df.loc[no_publicado, ["escalamiento_zona", "escalamiento_global"]] = np.nan
    df = df.drop_duplicates(subset=["zona", "año", "mes_num"])
    return df


# ── Incidentes por UGT ────────────────────────────────────────────────────────
def _incidentes_defensoria_hist() -> pd.DataFrame:
    """Protestas proxy de los Reportes Mensuales de la Defensoría (2018-2023).

    Cada mes-UGT con posible_protesta=True se convierte en un evento con
    fecha=día-15 del mes (midpoint mensual). Solo se incluyen eventos ANTES del
    inicio de dataIncidentes para evitar doble conteo. Meses sin PDF disponible
    se tratán implícitamente como sin protesta (limitación documentada)."""
    ruta = INTERIM_DIR / "defensoria_hist_conflictos.parquet"
    if not ruta.exists():
        return pd.DataFrame(columns=["fecha", "ugt", "categoria"])
    d = pd.read_parquet(ruta)
    # Un evento por (mes, UGT) aunque haya múltiples conflictos en el reporte
    prot = (
        d.groupby(["anio", "mes", "ugt"])["posible_protesta"]
        .any()
        .reset_index()
        .query("posible_protesta")
    )
    prot["fecha"] = pd.to_datetime(
        prot["anio"].astype(str) + "-" + prot["mes"].astype(str).str.zfill(2) + "-15"
    )
    prot = prot[prot["fecha"] < FECHA_INICIO_INCIDENTES].copy()
    prot["categoria"] = "PROTESTA"
    return prot[["fecha", "ugt", "categoria"]]


def _incidentes_por_ugt() -> pd.DataFrame:
    """Carga incidentes de protesta/violencia por UGT desde dataIncidentes +
    dataFormulario. Si FECHA_INICIO se extiende antes de FECHA_INICIO_INCIDENTES,
    añade los eventos proxy de la Defensoría histórica para llenar ese hueco.
    Con FECHA_INICIO=2024 los eventos Defensoría son pre-2024 y no aportan nada
    al label, pero SÍ cambian features de recencia (dias_desde_ultima_prot) de
    forma que reduce el PR-AUC de 0.807 a 0.779 — por eso se omiten cuando no
    se necesitan."""
    inc = _cargar_incidentes_raw()
    dist2ugt = {_norm(d): ugt for ugt, _, d in JERARQUIA}
    inc["ugt"] = inc["distrito_n"].map(dist2ugt)
    inc = inc[inc["ugt"].notna() & inc["categoria"].isin(["PROTESTA", "VIOLENCIA"])].copy()
    inc = inc[["fecha", "ugt", "categoria"]]

    if pd.Timestamp(FECHA_INICIO) < FECHA_INICIO_INCIDENTES:
        hist = _incidentes_defensoria_hist()
        if len(hist) > 0:
            inc = pd.concat([hist, inc], ignore_index=True).drop_duplicates(["fecha", "ugt", "categoria"])
    return inc


# ── Features autoregresivas por UGT ──────────────────────────────────────────
def _features_incidentes(skeleton: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    iw = inc.copy()
    iw["semana_inicio"] = _floor_to_monday(iw["fecha"])

    cat = iw.groupby(["ugt", "semana_inicio", "categoria"]).size().unstack(fill_value=0)
    for c in ["PROTESTA", "VIOLENCIA"]:
        if c not in cat.columns:
            cat[c] = 0
    cat = cat[["PROTESTA", "VIOLENCIA"]].rename(columns={"PROTESTA": "inc_prot_1w", "VIOLENCIA": "inc_viol_1w"}).reset_index()

    feat = skeleton.merge(cat, on=["ugt", "semana_inicio"], how="left")
    feat[["inc_prot_1w", "inc_viol_1w"]] = feat[["inc_prot_1w", "inc_viol_1w"]].fillna(0).astype(int)
    feat = feat.sort_values(["ugt", "semana_inicio"]).reset_index(drop=True)

    feat["inc_prot_4w"] = feat.groupby("ugt")["inc_prot_1w"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    feat["inc_viol_4w"] = feat.groupby("ugt")["inc_viol_1w"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    feat["delta_prot"] = feat.groupby("ugt")["inc_prot_1w"].transform(
        lambda x: x - x.shift(1).rolling(4, min_periods=1).mean()
    )
    feat["racha_prot"] = feat.groupby("ugt")["inc_prot_1w"].transform(_racha_consecutiva)
    feat["dias_desde_ultima_prot"] = _dias_desde_ultima(feat, inc[inc["categoria"] == "PROTESTA"])
    return feat


def _dias_desde_ultima(skeleton: pd.DataFrame, eventos: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=skeleton.index)
    for ugt in skeleton["ugt"].unique():
        ev = np.sort(eventos[eventos["ugt"] == ugt]["fecha"].values.astype("datetime64[ns]"))
        if len(ev) == 0:
            continue
        skel = skeleton[skeleton["ugt"] == ugt]
        sv = _fin_semana(skel["semana_inicio"].values)  # as-of domingo de cierre
        pos = np.searchsorted(ev, sv, side="right") - 1
        dias = np.full(len(sv), np.nan)
        ok = pos >= 0
        dias[ok] = (sv[ok].astype("int64") - ev[pos[ok]].astype("int64")) / 86_400_000_000_000
        result.loc[skel.index] = dias
    return result


# ── Label anti-fuga ───────────────────────────────────────────────────────────
def _labels(master: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    prot = inc[inc["categoria"] == "PROTESTA"]
    corte = np.datetime64(FECHA_CORTE, "ns")
    for h in HORIZONTES:
        col = f"y_{h}"
        master[col] = np.nan
        for ugt in UGTS:
            ev = np.sort(prot[prot["ugt"] == ugt]["fecha"].values.astype("datetime64[ns]"))
            idx = master[master["ugt"] == ugt].index
            sf = _fin_semana(master.loc[idx, "semana_inicio"].values)
            ini = sf + np.timedelta64(1, "D")   # lunes siguiente al cierre
            fin = sf + np.timedelta64(h, "D")   # cierre + h días (inclusive)
            if len(ev) > 0:
                hay = (np.searchsorted(ev, ini, "left") < np.searchsorted(ev, fin, "right")).astype(float)
            else:
                hay = np.zeros(len(sv))
            hay[fin > corte] = np.nan
            master.loc[idx, col] = hay
    return master


# ── Join calendario / INEI / Defensoría ──────────────────────────────────────
def _join_calendario(master: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_parquet(INTERIM_DIR / "calendario.parquet")
    cal["semana_inicio"] = pd.to_datetime(cal["semana_inicio"])
    cols = ["semana_inicio", "n_feriados", "es_semana_electoral", "es_fecha_critica",
            "dias_hasta_eleccion", "mes", "trimestre", "semana_iso"]
    return master.merge(cal[cols], on="semana_inicio", how="left")


def _join_inei(master: pd.DataFrame) -> pd.DataFrame:
    inei = pd.read_parquet(INTERIM_DIR / "inei_pobreza.parquet")
    pobreza_anc = inei[inei["zona"] == "Áncash"].copy()
    master = master.copy()
    master["_año"] = master["semana_inicio"].dt.year.clip(upper=2025)
    merged = master.merge(
        pobreza_anc.rename(columns={"año": "_año"})[["_año", "tasa_pobreza"]],
        on="_año", how="left",
    ).drop(columns="_año")
    return merged


def _join_defensoria(master: pd.DataFrame) -> pd.DataFrame:
    deff = _cargar_defensoria()
    if deff.empty:
        master["def_escalamiento_ancash"] = np.nan
        return master
    anc = deff[deff["zona"] == ZONA_DEFENSORIA][["año", "mes_num", "escalamiento_zona"]].copy()
    fecha_rep = master["semana_inicio"] - pd.DateOffset(months=2)  # rezago anti-fuga
    master = master.copy()
    master["_a"] = fecha_rep.dt.year
    master["_m"] = fecha_rep.dt.month
    merged = master.merge(
        anc.rename(columns={"año": "_a", "mes_num": "_m", "escalamiento_zona": "def_escalamiento_ancash"}),
        on=["_a", "_m"], how="left",
    ).drop(columns=["_a", "_m"])
    merged = merged.sort_values(["ugt", "semana_inicio"])
    merged["def_escalamiento_ancash"] = merged.groupby("ugt")["def_escalamiento_ancash"].ffill()
    return merged


def _join_defensoria_conflictos(master: pd.DataFrame) -> pd.DataFrame:
    """Cronicidad de conflictos por UGT desde los Reportes Mensuales de la
    Defensoría (src.data.loader_defensoria_historico). A diferencia de
    `def_escalamiento_ancash` (departamental, un solo valor para las 4 UGTs),
    esto es POR UGT y varía en el tiempo: cuántos conflictos activos/latentes
    tiene cada UGT cada mes, y cuántos están dirigidos a Antamina.

    Es feature de CONTEXTO (cronicidad), NO label — la mera existencia de un
    conflicto crónico no dispara el target (CLAUDE.md). Anti-fuga: rezago de
    2 meses (el reporte del mes M se publica en M+1) + ffill por UGT, que
    además cubre los meses-reporte que no se pudieron descargar."""
    ruta = INTERIM_DIR / "defensoria_hist_conflictos.parquet"
    cols = ["def_conf_activos_ugt", "def_conf_antamina_ugt"]
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Corre src.data.loader_defensoria_historico.")
        for c in cols:
            master[c] = np.nan
        return master
    d = pd.read_parquet(ruta)
    agg = d.groupby(["anio", "mes", "ugt"]).agg(
        def_conf_activos_ugt=("conflicto_activo", "sum"),
        def_conf_antamina_ugt=("es_antamina", "sum"),
    ).reset_index()

    fecha_rep = master["semana_inicio"] - pd.DateOffset(months=2)  # rezago anti-fuga
    master = master.copy()
    master["_a"] = fecha_rep.dt.year
    master["_m"] = fecha_rep.dt.month
    merged = master.merge(
        agg.rename(columns={"anio": "_a", "mes": "_m"}),
        on=["_a", "_m", "ugt"], how="left",
    ).drop(columns=["_a", "_m"])
    merged = merged.sort_values(["ugt", "semana_inicio"])
    for c in cols:
        merged[c] = merged.groupby("ugt")[c].ffill()
    return merged


def _join_oefa(master: pd.DataFrame) -> pd.DataFrame:
    """Tensión ambiental minera (denuncias OEFA/SINADA), contexto acumulado
    por UGT — ventana móvil de 12 meses. NO es un predictor de corto plazo
    validado: se intentó comprobar si las denuncias anteceden a las protestas
    reales y el resultado fue inconcluso por falta de superposición temporal
    entre fuentes (ver memoria del proyecto, project-oefa-denuncias). Se
    incluye como señal de contexto, igual que `def_escalamiento_ancash`."""
    ruta = INTERIM_DIR / "oefa_denuncias_ancash.parquet"
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Corre src.data.loader_oefa primero.")
        master["oefa_denuncias_mineria_12m"] = np.nan
        return master

    oefa = pd.read_parquet(ruta)
    dist2ugt = {_norm(d): ugt for ugt, _, d in JERARQUIA}
    oefa["distrito_n"] = oefa["distrito"].apply(_norm)
    oefa["ugt"] = oefa["distrito_n"].map(dist2ugt)
    es_mineria = oefa["actividad_economica"].str.upper().str.contains("MINER", na=False)
    oefa_min = oefa[oefa["ugt"].notna() & es_mineria][["fecha", "ugt"]]

    master = master.sort_values(["ugt", "semana_inicio"]).reset_index(drop=True)
    conteo = pd.Series(0, index=master.index, dtype=int)
    for ugt in UGTS:
        ev = np.sort(oefa_min[oefa_min["ugt"] == ugt]["fecha"].values.astype("datetime64[ns]"))
        idx = master[master["ugt"] == ugt].index
        sv = _fin_semana(master.loc[idx, "semana_inicio"].values)
        ini = sv - np.timedelta64(364, "D")  # ventana móvil de 12 meses (anti-fuga: solo hacia atrás)
        if len(ev) > 0:
            conteo.loc[idx] = (np.searchsorted(ev, sv, "right") - np.searchsorted(ev, ini, "left")).astype(int)
    master["oefa_denuncias_mineria_12m"] = conteo
    return master


def _join_reportes(master: pd.DataFrame) -> pd.DataFrame:
    """Tensión y compromiso específicos de Antamina, desde los reportes
    situacionales semanales/mensuales de Antamina (Word/PowerPoint,
    `src/data/loader_reportes.py`). A diferencia de `inc_*` (cualquier
    conflicto en la zona, sin importar el objetivo), aquí cada párrafo se
    etiquetó con si menciona a Antamina explícitamente — la única fuente del
    proyecto con esa distinción. Ventana móvil de 4 semanas, anti-fuga (solo
    hacia atrás)."""
    ruta = INTERIM_DIR / "reportes_antamina_parrafos.parquet"
    cols_salida = ["rep_antamina_neg_4w", "rep_compromiso_4w"]
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Corre src.data.loader_reportes primero.")
        for c in cols_salida:
            master[c] = np.nan
        return master

    rep = pd.read_parquet(ruta)
    rep["fecha_reporte"] = pd.to_datetime(rep["fecha_reporte"])
    # Vigencia de cada párrafo: 28 días (ventana de 4 semanas, reportes
    # semanales) o el largo del periodo si es un reporte mensual ("09.06.26 al
    # 09.07.26", ~31 días), para que las semanas entre dos mensuales no caigan a 0.
    vig = rep["vigencia_dias"] if "vigencia_dias" in rep.columns else pd.Series(28, index=rep.index)
    rep["vigencia_dias"] = vig.fillna(28).clip(lower=28).astype(int)
    neg = rep[rep["menciona_antamina"] & rep["tono_negativo"]]
    compromiso = rep[rep["tono_compromiso"]]

    master = master.sort_values(["ugt", "semana_inicio"]).reset_index(drop=True)
    out = {c: np.zeros(len(master), dtype=int) for c in cols_salida}
    for ugt in UGTS:
        idx = master[master["ugt"] == ugt].index
        sv = _fin_semana(master.loc[idx, "semana_inicio"].values)  # as-of domingo
        for col, fuente in [("rep_antamina_neg_4w", neg), ("rep_compromiso_4w", compromiso)]:
            f = fuente[fuente["ugt"] == ugt]
            if f.empty:
                continue
            ev = f["fecha_reporte"].values.astype("datetime64[ns]")
            hasta = ev + f["vigencia_dias"].values.astype("timedelta64[D]")
            # párrafo vigente en sv: publicado <= sv <= publicado + vigencia
            vigente = (ev[None, :] <= sv[:, None]) & (sv[:, None] <= hasta[None, :])
            out[col][idx] = vigente.sum(axis=1).astype(int)
    for c in cols_salida:
        master[c] = out[c]
    return master


def _join_cobre(master: pd.DataFrame) -> pd.DataFrame:
    """Precio del cobre (HG=F, COMEX) como feature macroeconómica global.

    Dos features, ambas con anti-fuga (se usa el cierre de la semana anterior
    a semana_inicio, nunca el de la semana en curso):
      - cobre_precio_usd : cierre de la semana t-1 en USD/lb
      - cobre_ret_4w     : retorno porcentual de t-5 a t-1 (momentum 4 semanas)

    Feature NO específica por UGT — el precio del cobre es el mismo para las
    4 UGTs en cada semana (señal macroeconómica global, como tasa_pobreza).

    Hipótesis: precio alto → comunidades exigen más regalías/beneficios;
    caída brusca → recortes de inversión social. Ambas pueden disparar tensión.
    Se incluye el nivel Y el momentum para capturar ambas direcciones."""
    ruta = INTERIM_DIR / "cobre_semanal.parquet"
    cols_salida = ["cobre_precio_usd", "cobre_ret_4w"]
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Corre src.data.loader_cobre primero.")
        for c in cols_salida:
            master[c] = np.nan
        return master

    cobre = pd.read_parquet(ruta)
    fechas = cobre["fecha"].values.astype("datetime64[ns]")
    precios = cobre["cobre_usd_lb"].values.astype(float)

    sv = master["semana_inicio"].values.astype("datetime64[ns]")
    # Posición de la semana ANTERIOR (última observación estrictamente antes de semana_inicio)
    pos_prev = np.searchsorted(fechas, sv, side="left") - 1
    # Posición de 4 semanas antes del cierre anterior (aprox. 28 días)
    ventana_4w = sv - np.timedelta64(28, "D")
    pos_4w = np.searchsorted(fechas, ventana_4w, side="left") - 1

    precio_prev = np.where(pos_prev >= 0, precios[np.clip(pos_prev, 0, len(precios) - 1)], np.nan)
    precio_4w   = np.where(pos_4w  >= 0, precios[np.clip(pos_4w,  0, len(precios) - 1)], np.nan)

    master = master.copy()
    master["cobre_precio_usd"] = precio_prev
    with np.errstate(divide="ignore", invalid="ignore"):
        master["cobre_ret_4w"] = np.where(
            precio_4w > 0,
            (precio_prev - precio_4w) / precio_4w * 100,
            np.nan,
        )
    return master


def _join_historica(master: pd.DataFrame) -> pd.DataFrame:
    """Tensión histórica acumulada por UGT desde el inicio de operaciones
    de ANTAMINA (2001), extraída de la fuente de prensa escrita.

    Dos features, ambas con anti-fuga estricta (solo eventos previos a
    semana_inicio):
      - hist_prot_antamina_acum : total acumulado desde 2001
      - hist_prot_antamina_5y   : ventana móvil de 5 años (260 semanas)

    No extiende la ventana de entrenamiento (dataIncidentes cubre solo desde
    2023-08, por lo que las features autoregresivas serían NaN para períodos
    anteriores). Aporta información estructural de largo plazo: qué UGT ha
    tenido históricamente más conflictos dirigidos a Antamina, capturando
    diferencias entre UGTs que las features de corto plazo no ven."""
    ruta = INTERIM_DIR / "historica_protestas_ugt.parquet"
    cols_salida = ["hist_prot_antamina_acum", "hist_prot_antamina_5y"]
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Corre src.data.loader_historica primero.")
        for c in cols_salida:
            master[c] = np.nan
        return master

    hist = pd.read_parquet(ruta)
    hist = hist[hist["es_antamina"]].copy()
    hist["fecha"] = pd.to_datetime(hist["fecha"])

    master = master.sort_values(["ugt", "semana_inicio"]).reset_index(drop=True)
    acum = np.zeros(len(master), dtype=int)
    mov5y = np.zeros(len(master), dtype=int)
    for ugt in UGTS:
        ev = np.sort(hist[hist["ugt"] == ugt]["fecha"].values.astype("datetime64[ns]"))
        if len(ev) == 0:
            continue
        idx = master[master["ugt"] == ugt].index
        sv = _fin_semana(master.loc[idx, "semana_inicio"].values)
        # acumulado: todos los eventos estrictamente anteriores al cierre
        acum[idx] = np.searchsorted(ev, sv, side="left")
        # ventana móvil 5 años (1820 días ≈ 260 semanas)
        ini5 = sv - np.timedelta64(1820, "D")
        mov5y[idx] = (np.searchsorted(ev, sv, "left") - np.searchsorted(ev, ini5, "left")).astype(int)
    master["hist_prot_antamina_acum"] = acum
    master["hist_prot_antamina_5y"] = mov5y
    return master


def _cargar_protestas_impacto() -> pd.DataFrame:
    """Nivel de impacto (severidad, 70-500) de dataProtestas.xlsx, dentro de
    las 4 UGTs. Es un subconjunto de los mismos eventos de `_incidentes_por_ugt`
    (87% se solapan) con un dato que esa fuente no tiene: severidad por evento,
    no solo conteo. No se usa como label, solo como feature de contexto."""
    ruta = Path("src/data") / "dataProtestas.xlsx"
    if not ruta.exists():
        return pd.DataFrame(columns=["fecha", "ugt", "nivel_impacto"])
    raw = pd.read_excel(ruta, sheet_name="Hoja1")
    dist2ugt = {_norm(d): ugt for ugt, _, d in JERARQUIA}
    df = pd.DataFrame()
    df["fecha"] = pd.to_datetime(raw["ID_FECHA"], errors="coerce")
    df["ugt"] = raw["DISTRITO"].astype(str).apply(_norm).map(dist2ugt)
    df["nivel_impacto"] = pd.to_numeric(raw["NIVEL DE IMPACTO"], errors="coerce")
    return df[df["ugt"].notna() & df["fecha"].notna() & df["nivel_impacto"].notna()]


def _join_impacto(master: pd.DataFrame) -> pd.DataFrame:
    """Severidad (nivel de impacto) de la última protesta conocida por UGT,
    forward-fill — análogo a `dias_desde_ultima_prot` pero de severidad en
    vez de recencia. Anti-fuga: solo mira el evento más reciente ANTES de la
    semana actual."""
    impacto = _cargar_protestas_impacto()
    master = master.sort_values(["ugt", "semana_inicio"]).reset_index(drop=True)
    valor = pd.Series(np.nan, index=master.index)
    for ugt in UGTS:
        ev = impacto[impacto["ugt"] == ugt].sort_values("fecha")
        ev_fecha = ev["fecha"].values.astype("datetime64[ns]")
        ev_valor = ev["nivel_impacto"].values
        idx = master[master["ugt"] == ugt].index
        sv = master.loc[idx, "semana_inicio"].values.astype("datetime64[ns]")
        pos = np.searchsorted(ev_fecha, sv, side="left") - 1
        v = np.full(len(sv), np.nan)
        ok = pos >= 0
        v[ok] = ev_valor[pos[ok]]
        valor.loc[idx] = v
    master["prot_impacto_ultimo"] = valor
    return master


# ── Principal ─────────────────────────────────────────────────────────────────
def construir(fecha_corte: pd.Timestamp | None = None) -> pd.DataFrame:
    """Tabla maestra UGT × semana desde FECHA_INICIO hasta la última semana
    COMPLETA (domingo de cierre <= fecha de corte + 1 día, para aceptar un
    corte en sábado). Las semanas recientes quedan con label NaN (aún no
    observable) y sirven para el scoring en vivo."""
    global FECHA_CORTE
    FECHA_CORTE = pd.Timestamp(fecha_corte or _fecha_corte_datos()).normalize()
    ultimo_lunes = FECHA_CORTE + pd.Timedelta(days=1) - pd.Timedelta(days=6)
    semanas = pd.date_range(FECHA_INICIO, ultimo_lunes, freq="W-MON")
    skeleton = pd.DataFrame(list(product(UGTS, semanas)), columns=["ugt", "semana_inicio"])

    inc = _incidentes_por_ugt()
    inc = inc[inc["fecha"] <= FECHA_CORTE + pd.Timedelta(days=1)]  # anti-fuga defensivo
    master = _features_incidentes(skeleton, inc)
    master = _join_calendario(master)
    master = _join_inei(master)
    master = _join_defensoria(master)
    master = _join_defensoria_conflictos(master)
    master = _join_oefa(master)
    master = _join_cobre(master)
    master = _join_impacto(master)
    master = _join_reportes(master)
    master = _join_historica(master)
    master = _labels(master, inc)

    master["semana_fin"] = master["semana_inicio"] + pd.Timedelta(days=6)
    master["fecha_corte_datos"] = FECHA_CORTE
    # Se conservan TODAS las semanas completas; las que no tienen label
    # observable quedan con y_h = NaN (entrenamiento las descarta, scoring las usa).
    cols_count = [c for c in master.columns if c.startswith("inc_") or c.startswith("delta_") or c == "racha_prot"]
    master[cols_count] = master[cols_count].fillna(0)
    # prot_impacto_ultimo: NaN real (no hubo protesta previa todavía) -> 0,
    # distinto de "hubo protesta de bajo impacto" (valor mínimo real es 70)
    master["prot_impacto_ultimo"] = master["prot_impacto_ultimo"].fillna(0)
    master["hist_prot_antamina_acum"] = master["hist_prot_antamina_acum"].fillna(0)
    master["hist_prot_antamina_5y"] = master["hist_prot_antamina_5y"].fillna(0)
    # Cronicidad Defensoría por UGT: NaN (sin reporte disponible aún) -> 0
    master["def_conf_activos_ugt"] = master["def_conf_activos_ugt"].fillna(0)
    master["def_conf_antamina_ugt"] = master["def_conf_antamina_ugt"].fillna(0)
    return master.reset_index(drop=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master = construir()
    master.to_parquet(SALIDA, index=False)
    master.to_csv(SALIDA.with_suffix(".csv"), index=False)

    print(f"✓ Dataset Áncash (UGT × semana) construido: {len(master)} filas, {master['ugt'].nunique()} UGTs")
    print(f"  Corte de datos: {FECHA_CORTE.date()}")
    print(f"  Rango (domingo de cierre): {master['semana_fin'].min().date()} → {master['semana_fin'].max().date()}")
    for h in HORIZONTES:
        sub = master[f"y_{h}"].dropna()
        print(f"  y_{h}: {sub.mean()*100:.1f}% positivos ({int(sub.sum())}/{len(sub)})")
    print(f"\n  Balance y_30 por UGT:")
    print(master.groupby("ugt")["y_30"].apply(lambda s: f"{s.mean()*100:.0f}% ({int(s.sum())}/{s.notna().sum()})"))


if __name__ == "__main__":
    main()
