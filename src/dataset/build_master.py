import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────
ZONAS: dict[str, str] = {
    "Áncash":          "A",
    "Huánuco":         "A",
    "Pasco":           "A",
    "Cajamarca":       "A",
    "La Libertad":     "A",
    "Ica":             "B",
    "Pisco":           "B",
    "Huarmey":         "B",
    "Barranca":        "B",
    "Lima Provincias": "B",
}

HORIZONTES = [7, 14, 30, 60]
FECHA_CORTE = pd.Timestamp.today().normalize()

# Zonas sub-departamentales sin reporte propio (Defensoría e incidentes
# reportan a nivel de departamento/provincia) → usar el depto/zona padre como proxy
ZONA_A_DEPTO_PADRE: dict[str, str] = {
    "Pisco":    "Ica",
    "Huarmey":  "Áncash",
    "Barranca": "Lima Provincias",
}

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


# ── Carga ─────────────────────────────────────────────────────────────────────
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
    df = df.drop_duplicates(subset=["zona", "año", "mes_num"])
    return df


def _cargar_incidentes() -> pd.DataFrame:
    ruta = INTERIM_DIR / "incidentes_normalizados.parquet"
    if not ruta.exists():
        warnings.warn(f"No se encontró {ruta}. Features de incidentes serán 0.")
        return pd.DataFrame(columns=["zona", "fecha", "categoria"])
    df = pd.read_parquet(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _cargar_fuentes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alertas = pd.read_parquet(INTERIM_DIR / "alertas_normalizadas.parquet")
    alertas["fecha"] = pd.to_datetime(alertas["fecha"])

    calendario = pd.read_parquet(INTERIM_DIR / "calendario.parquet")
    calendario["semana_inicio"] = pd.to_datetime(calendario["semana_inicio"])

    inei = pd.read_parquet(INTERIM_DIR / "inei_pobreza.parquet")
    defensoria = _cargar_defensoria()
    incidentes = _cargar_incidentes()
    return alertas, calendario, inei, defensoria, incidentes


# ── Paso 1: Esqueleto zona × semana ──────────────────────────────────────────
def _crear_skeleton(calendario: pd.DataFrame) -> pd.DataFrame:
    sk = pd.DataFrame(
        list(product(ZONAS.keys(), calendario["semana_inicio"])),
        columns=["zona", "semana_inicio"],
    )
    sk["track"] = sk["zona"].map(ZONAS)
    return sk


# ── Paso 2: Join calendario ────────────────────────────────────────────────────
def _join_calendario(master: pd.DataFrame, calendario: pd.DataFrame) -> pd.DataFrame:
    cal_cols = [c for c in calendario.columns if c != "semana_inicio"]
    return master.merge(calendario[["semana_inicio"] + cal_cols], on="semana_inicio", how="left")


# ── Paso 3: Features de alertas ───────────────────────────────────────────────
def _floor_to_monday(fechas: pd.Series) -> pd.Series:
    return fechas.dt.normalize() - pd.to_timedelta(fechas.dt.weekday, unit="D")


def _racha_consecutiva(s: pd.Series) -> pd.Series:
    """Semanas consecutivas con al menos 1 alerta (se reinicia en 0 al no haber)."""
    tiene = (s > 0).astype(int)
    grupos = (tiene != tiene.shift().fillna(0)).cumsum()
    return tiene.groupby(grupos).cumsum()


def _dias_desde_ultima_crit(skeleton: pd.DataFrame, alertas: pd.DataFrame) -> pd.Series:
    """Días desde la última alerta CRIT antes de semana_inicio, por zona."""
    crit = alertas[alertas["nivel"] == "CRIT"][["zona", "fecha"]].copy()
    result = pd.Series(np.nan, index=skeleton.index)

    for zona in skeleton["zona"].unique():
        az = crit[crit["zona"] == zona]["fecha"].sort_values()
        if az.empty:
            continue
        az_np = az.values.astype("datetime64[ns]")
        skel_z = skeleton[skeleton["zona"] == zona]
        sv_np = skel_z["semana_inicio"].values.astype("datetime64[ns]")

        pos = np.searchsorted(az_np, sv_np, side="left") - 1
        dias = np.full(len(sv_np), np.nan)
        valid = pos >= 0
        dias[valid] = (
            (sv_np[valid].astype("int64") - az_np[pos[valid]].astype("int64"))
            / 86_400_000_000_000
        )
        result.loc[skel_z.index] = dias

    return result


def _features_alertas(skeleton: pd.DataFrame, alertas: pd.DataFrame) -> pd.DataFrame:
    aw = alertas.copy()
    aw["semana_inicio"] = _floor_to_monday(aw["fecha"])

    # Conteos semanales por nivel
    niveles = ["CRIT", "ALRT", "INFO"]
    niv = aw.groupby(["zona", "semana_inicio", "nivel"]).size().unstack(fill_value=0)
    for n in niveles:
        if n not in niv.columns:
            niv[n] = 0
    niv = niv[niveles].rename(columns=lambda c: f"n_{c.lower()}_1w")

    # Conteos semanales por categoría
    categorias = ["PROTESTA", "VIOLENCIA", "POLITICA", "ECONOMIA", "AMBIENTAL"]
    cat = aw.groupby(["zona", "semana_inicio", "categoria"]).size().unstack(fill_value=0)
    for c in categorias:
        if c not in cat.columns:
            cat[c] = 0
    cat = cat[categorias].rename(columns=lambda c: f"n_{c.lower()}_1w")

    total = aw.groupby(["zona", "semana_inicio"]).size().rename("n_total_1w")

    weekly = niv.join(cat).join(total).reset_index()

    feat = skeleton[["zona", "semana_inicio"]].merge(weekly, on=["zona", "semana_inicio"], how="left")
    count_cols = [c for c in feat.columns if c.endswith("_1w")]
    feat[count_cols] = feat[count_cols].fillna(0).astype(int)

    # Ordenar por zona+semana para que rolling sea temporal
    feat = feat.sort_values(["zona", "semana_inicio"]).reset_index(drop=True)

    # Ventanas de 4 semanas (incluye la semana actual)
    feat["n_crit_4w"] = feat.groupby("zona")["n_crit_1w"].transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )
    feat["n_protesta_4w"] = feat.groupby("zona")["n_protesta_1w"].transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )

    # Aceleración respecto a las 4 semanas estrictamente previas
    feat["delta_crit"] = feat.groupby("zona")["n_crit_1w"].transform(
        lambda x: x - x.shift(1).rolling(4, min_periods=1).mean()
    )
    feat["delta_protesta"] = feat.groupby("zona")["n_protesta_1w"].transform(
        lambda x: x - x.shift(1).rolling(4, min_periods=1).mean()
    )

    # Racha de semanas consecutivas con al menos una alerta
    feat["racha_semanas_con_alerta"] = feat.groupby("zona")["n_total_1w"].transform(
        _racha_consecutiva
    )

    # Días desde la última alerta CRIT antes de esta semana
    feat["dias_desde_ultima_crit"] = _dias_desde_ultima_crit(feat, alertas)

    return feat


# ── Paso 3b: Features de incidentes (BD_Incidentes interna) ─────────────────
def _features_incidentes(skeleton: pd.DataFrame, incidentes: pd.DataFrame) -> pd.DataFrame:
    """Conteos semanales de protesta/violencia desde la base de incidentes
    geolocalizados interna. Complementa a alertas_propias: mayor densidad
    semanal en Áncash/Ica/Cajamarca, sin reemplazar la fuente existente."""
    cols_salida = ["inc_n_protesta_1w", "inc_n_violencia_1w", "inc_n_protesta_4w"]

    if incidentes.empty:
        feat = skeleton[["zona", "semana_inicio"]].copy()
        for col in cols_salida:
            feat[col] = 0
        return feat

    iw = incidentes.copy()
    iw["semana_inicio"] = _floor_to_monday(iw["fecha"])

    categorias = ["PROTESTA", "VIOLENCIA"]
    cat = iw.groupby(["zona", "semana_inicio", "categoria"]).size().unstack(fill_value=0)
    for c in categorias:
        if c not in cat.columns:
            cat[c] = 0
    cat = cat[categorias].rename(columns=lambda c: f"inc_n_{c.lower()}_1w").reset_index()

    # Zonas sub-departamentales (Pisco/Huarmey/Barranca) no tienen incidentes
    # propios geolocalizados: usamos los de su zona padre como proxy, igual
    # que con Defensoría.
    skel_lookup = skeleton[["zona", "semana_inicio"]].copy()
    skel_lookup["_zona_padre"] = skel_lookup["zona"].map(ZONA_A_DEPTO_PADRE).fillna(skel_lookup["zona"])

    feat = skel_lookup.merge(
        cat.rename(columns={"zona": "_zona_padre"}), on=["_zona_padre", "semana_inicio"], how="left"
    ).drop(columns=["_zona_padre"])
    count_cols = [c for c in feat.columns if c.endswith("_1w")]
    feat[count_cols] = feat[count_cols].fillna(0).astype(int)

    feat = feat.sort_values(["zona", "semana_inicio"]).reset_index(drop=True)
    feat["inc_n_protesta_4w"] = feat.groupby("zona")["inc_n_protesta_1w"].transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )

    return feat[["zona", "semana_inicio"] + cols_salida]


# ── Paso 4: Features de Defensoría ───────────────────────────────────────────
def _features_defensoria(master: pd.DataFrame, defensoria: pd.DataFrame) -> pd.DataFrame:
    def_cols = ["def_escalamiento_zona", "def_escalamiento_global"]

    if defensoria.empty:
        for col in def_cols:
            master[col] = np.nan
        return master

    # Anti-fuga: reporte del mes M publicado en M+1, usamos rezago de 2 meses para seguridad
    fecha_rep = master["semana_inicio"] - pd.DateOffset(months=2)
    master = master.copy()
    master["_año_def"] = fecha_rep.dt.year
    master["_mes_def"] = fecha_rep.dt.month
    # Zonas sub-departamentales (Pisco, Huarmey, Barranca) no tienen reporte propio
    # en Defensoría: usamos el escalamiento de su departamento padre como proxy.
    master["_zona_def"] = master["zona"].map(ZONA_A_DEPTO_PADRE).fillna(master["zona"])

    def_prep = (
        defensoria
        .rename(columns={
            "escalamiento_zona":   "def_escalamiento_zona",
            "escalamiento_global": "def_escalamiento_global",
            "zona":                "_zona_def_k",
            "año":                 "_año_def_k",
            "mes_num":             "_mes_def_k",
        })
        [["_zona_def_k", "_año_def_k", "_mes_def_k", "def_escalamiento_zona", "def_escalamiento_global"]]
    )

    merged = master.merge(
        def_prep,
        left_on=["_zona_def", "_año_def", "_mes_def"],
        right_on=["_zona_def_k", "_año_def_k", "_mes_def_k"],
        how="left",
    ).drop(columns=["_zona_def", "_año_def", "_mes_def", "_zona_def_k", "_año_def_k", "_mes_def_k"])

    # Forward fill por zona (propaga el último reporte disponible)
    merged = merged.sort_values(["zona", "semana_inicio"])
    for col in def_cols:
        merged[col] = merged.groupby("zona")[col].ffill()

    return merged


# ── Paso 5: INEI ──────────────────────────────────────────────────────────────
def _join_inei(master: pd.DataFrame, inei: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    # Para 2026 usar el dato de 2025 (último publicado disponible)
    master["_año_inei"] = master["año"].clip(upper=2025)
    merged = master.merge(
        inei.rename(columns={"año": "_año_inei"}),
        on=["zona", "_año_inei"],
        how="left",
    ).drop(columns=["_año_inei"])
    return merged


# ── Paso 6: Etiquetas (CRÍTICO: sin fuga) ────────────────────────────────────
def _labels_track_b(master: pd.DataFrame, alertas: pd.DataFrame) -> pd.DataFrame:
    """y_h = 1 si hay alerta CRIT o PROTESTA en (semana_inicio, semana_inicio+h]."""
    alertas_rel = alertas[
        (alertas["nivel"] == "CRIT") | (alertas["categoria"] == "PROTESTA")
    ].copy()
    fecha_corte_np = np.datetime64(FECHA_CORTE)

    for h in HORIZONTES:
        col = f"y_{h}"
        for zona in master.loc[master["track"] == "B", "zona"].unique():
            az = (
                alertas_rel[alertas_rel["zona"] == zona]["fecha"]
                .sort_values().values.astype("datetime64[ns]")
            )
            idx_z = master[(master["track"] == "B") & (master["zona"] == zona)].index
            sv = master.loc[idx_z, "semana_inicio"].values.astype("datetime64[ns]")

            inicio = sv + np.timedelta64(1, "D")
            fin    = sv + np.timedelta64(h, "D")

            if len(az) > 0:
                l = np.searchsorted(az, inicio, side="left")
                r = np.searchsorted(az, fin,   side="right")
                hay = (r > l).astype(float)
            else:
                hay = np.zeros(len(sv), dtype=float)

            hay[fin > fecha_corte_np] = np.nan
            master.loc[idx_z, col] = hay

    return master


def _labels_track_a(master: pd.DataFrame, defensoria: pd.DataFrame) -> pd.DataFrame:
    """y_h = 1 si el reporte de Defensoría del mes que cubre semana_inicio+h muestra
    escalamiento_zona > mes anterior (actividad nueva en esa zona)."""
    mask_a = master["track"] == "A"

    if defensoria.empty:
        for h in HORIZONTES:
            master.loc[mask_a, f"y_{h}"] = np.nan
        return master

    # Precomputar flag de novedad mes a mes por zona
    ds = defensoria.sort_values(["zona", "año", "mes_num"]).copy()
    ds["escal_zona_prev"] = ds.groupby("zona")["escalamiento_zona"].shift(1)
    ds["hay_novedad"] = (
        ds["escalamiento_zona"].fillna(0) > ds["escal_zona_prev"].fillna(0)
    ).astype(float)
    # NaN para meses sin datos fiables (esc_global == 0 indica datos no publicados aún)
    ds.loc[ds["escalamiento_global"] == 0, "hay_novedad"] = np.nan
    def_nov = ds[["zona", "año", "mes_num", "hay_novedad"]].copy()

    tmp = master[mask_a].copy()

    for h in HORIZONTES:
        col = f"y_{h}"
        fecha_futura = tmp["semana_inicio"] + pd.Timedelta(days=h)
        tmp_h = tmp[["zona", "semana_inicio"]].copy()
        tmp_h["_año_lab"] = fecha_futura.dt.year
        tmp_h["_mes_lab"] = fecha_futura.dt.month

        merged = tmp_h.merge(
            def_nov.rename(columns={"año": "_año_lab", "mes_num": "_mes_lab"}),
            on=["zona", "_año_lab", "_mes_lab"],
            how="left",
        )
        # Left join preserva orden de tmp_h; restaurar índice original
        merged.index = tmp.index

        labels = merged["hay_novedad"].copy()
        labels[fecha_futura > FECHA_CORTE] = np.nan
        master.loc[tmp.index, col] = labels.values

    return master


# ── Paso 7: Limpieza ──────────────────────────────────────────────────────────
def _limpiar(master: pd.DataFrame) -> pd.DataFrame:
    # Eliminar las semanas más recientes donde y_60 aún no es computable
    corte = FECHA_CORTE - pd.Timedelta(days=60)
    master = master[master["semana_inicio"] <= corte].copy()

    # Rellenar NaN en features de conteo con 0 (NaN = sin alertas/incidentes esa semana)
    cols_count = [
        c for c in master.columns
        if (c.startswith("n_") or c.startswith("inc_") or c.startswith("delta_") or c == "racha_semanas_con_alerta")
        and c not in [f"y_{h}" for h in HORIZONTES]
    ]
    master[cols_count] = master[cols_count].fillna(0)

    return master.reset_index(drop=True)


# ── Paso 8: Guardar y reportar ─────────────────────────────────────────────────
def _guardar_y_reportar(master: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(PROCESSED_DIR / "master_zona_semana.parquet", index=False)
    master.to_csv(PROCESSED_DIR / "master_zona_semana.csv", index=False)

    def balance(mask: pd.Series, col: str) -> tuple[str, int]:
        sub = master.loc[mask, col].dropna()
        if sub.empty:
            return "sin datos", 0
        return f"{round(sub.mean() * 100, 1)}% positivos", len(sub)

    mask_a = master["track"] == "A"
    mask_b = master["track"] == "B"
    bal_a, n_a = balance(mask_a, "y_30")
    bal_b, n_b = balance(mask_b, "y_30")

    label_cols = {f"y_{h}" for h in HORIZONTES}
    cols_nan = [
        c for c in master.columns
        if c not in label_cols and master[c].isna().mean() > 0.20
    ]

    print(f"""
✓ Tabla maestra construida:
  Filas totales:     {len(master):,}
  Zonas:             {master['zona'].nunique()}
  Rango semanas:     {master['semana_inicio'].min().date()} → {master['semana_inicio'].max().date()}
  Columnas:          {len(master.columns)}

  Balance de etiquetas (y_30):
    Track A: {bal_a} ({n_a} filas con label)
    Track B: {bal_b} ({n_b} filas con label)

  Features con >20% NaN: {cols_nan if cols_nan else 'ninguna'}

  Muestra (zona, semana, alertas, defensoría, pobreza, labels):
{master[['zona','semana_inicio','n_crit_1w','n_protesta_1w','def_escalamiento_zona','tasa_pobreza','y_7','y_30']].head(4).to_string()}
""")


# ── Principal ─────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando fuentes...")
    alertas, calendario, inei, defensoria, incidentes = _cargar_fuentes()

    print("Paso 1: esqueleto zona × semana...")
    master = _crear_skeleton(calendario)

    print("Paso 2: join calendario...")
    master = _join_calendario(master, calendario)

    print("Paso 3: features de alertas...")
    feat_alertas = _features_alertas(master[["zona", "semana_inicio"]].copy(), alertas)
    alert_cols = [c for c in feat_alertas.columns if c not in ["zona", "semana_inicio"]]
    master = master.merge(
        feat_alertas[["zona", "semana_inicio"] + alert_cols],
        on=["zona", "semana_inicio"],
        how="left",
    )

    print("Paso 3b: features de incidentes (BD_Incidentes interna)...")
    feat_incidentes = _features_incidentes(master[["zona", "semana_inicio"]].copy(), incidentes)
    inc_cols = [c for c in feat_incidentes.columns if c not in ["zona", "semana_inicio"]]
    master = master.merge(
        feat_incidentes[["zona", "semana_inicio"] + inc_cols],
        on=["zona", "semana_inicio"],
        how="left",
    )

    print("Paso 4: features de Defensoría...")
    master = _features_defensoria(master, defensoria)

    print("Paso 5: INEI...")
    master = _join_inei(master, inei)

    print("Paso 6: etiquetas...")
    for h in HORIZONTES:
        master[f"y_{h}"] = np.nan
    master = _labels_track_b(master, alertas)
    master = _labels_track_a(master, defensoria)

    print("Paso 7: limpieza...")
    master = _limpiar(master)

    print("Paso 8: guardando...")
    _guardar_y_reportar(master)


if __name__ == "__main__":
    main()
