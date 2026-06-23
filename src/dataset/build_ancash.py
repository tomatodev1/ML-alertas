"""Dataset maestro Track A regional — Áncash, unidad = UGT × semana.

Reactiva Track A usando la BD interna de incidentes como fuente de label con
resolución SEMANAL (lo que Defensoría, mensual, no permitía). La unidad es la
UGT de ANTAMINA (Mina San Marcos, Huallanca, Valle Fortaleza, Huarmey), que es
la unidad operativa real del cliente y da un label no trivial.

Anti-fuga: features de la semana t usan datos hasta el fin de t; el label y_h
mira la ventana futura [t+1, t+h].
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.build_master import (
    _cargar_defensoria,
    _floor_to_monday,
    _racha_consecutiva,
    ZONA_A_DEPTO_PADRE,  # no se usa aquí pero documenta el patrón de proxy
)
from src.scoring.ancash_datos import JERARQUIA, _cargar_incidentes_raw, _norm

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
SALIDA = PROCESSED_DIR / "master_ancash_ugt.parquet"

UGTS = ["Mina San Marcos", "Huallanca", "Valle Fortaleza", "Huarmey"]
HORIZONTES = [14, 30, 60]
FECHA_INICIO = "2024-01-01"
FECHA_CORTE = pd.Timestamp("2026-04-13")  # mismo corte que la tabla maestra principal

ZONA_DEFENSORIA = "Áncash"  # las 4 UGTs comparten el contexto departamental


# ── Incidentes por UGT ────────────────────────────────────────────────────────
def _incidentes_por_ugt() -> pd.DataFrame:
    inc = _cargar_incidentes_raw()
    dist2ugt = {_norm(d): ugt for ugt, _, d in JERARQUIA}
    inc["ugt"] = inc["distrito_n"].map(dist2ugt)
    inc = inc[inc["ugt"].notna() & inc["categoria"].isin(["PROTESTA", "VIOLENCIA"])].copy()
    return inc[["fecha", "ugt", "categoria"]]


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
        sv = skel["semana_inicio"].values.astype("datetime64[ns]")
        pos = np.searchsorted(ev, sv, side="left") - 1
        dias = np.full(len(sv), np.nan)
        ok = pos >= 0
        dias[ok] = (sv[ok].astype("int64") - ev[pos[ok]].astype("int64")) / 86_400_000_000_000
        result.loc[skel.index] = dias
    return result


# ── Label anti-fuga ───────────────────────────────────────────────────────────
def _labels(master: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    prot = inc[inc["categoria"] == "PROTESTA"]
    corte = np.datetime64(FECHA_CORTE)
    for h in HORIZONTES:
        col = f"y_{h}"
        master[col] = np.nan
        for ugt in UGTS:
            ev = np.sort(prot[prot["ugt"] == ugt]["fecha"].values.astype("datetime64[ns]"))
            idx = master[master["ugt"] == ugt].index
            sv = master.loc[idx, "semana_inicio"].values.astype("datetime64[ns]")
            ini = sv + np.timedelta64(1, "D")
            fin = sv + np.timedelta64(h, "D")
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


# ── Principal ─────────────────────────────────────────────────────────────────
def construir() -> pd.DataFrame:
    semanas = pd.date_range(FECHA_INICIO, FECHA_CORTE, freq="W-MON")
    skeleton = pd.DataFrame(list(product(UGTS, semanas)), columns=["ugt", "semana_inicio"])

    inc = _incidentes_por_ugt()
    master = _features_incidentes(skeleton, inc)
    master = _join_calendario(master)
    master = _join_inei(master)
    master = _join_defensoria(master)
    master = _labels(master, inc)

    # Limpieza: descartar semanas sin label y_60 computable; rellenar conteos
    master = master[master["semana_inicio"] <= FECHA_CORTE - pd.Timedelta(days=60)].copy()
    cols_count = [c for c in master.columns if c.startswith("inc_") or c.startswith("delta_") or c == "racha_prot"]
    master[cols_count] = master[cols_count].fillna(0)
    return master.reset_index(drop=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master = construir()
    master.to_parquet(SALIDA, index=False)
    master.to_csv(SALIDA.with_suffix(".csv"), index=False)

    print(f"✓ Dataset Áncash (UGT × semana) construido: {len(master)} filas, {master['ugt'].nunique()} UGTs")
    print(f"  Rango: {master['semana_inicio'].min().date()} → {master['semana_inicio'].max().date()}")
    for h in HORIZONTES:
        sub = master[f"y_{h}"].dropna()
        print(f"  y_{h}: {sub.mean()*100:.1f}% positivos ({int(sub.sum())}/{len(sub)})")
    print(f"\n  Balance y_30 por UGT:")
    print(master.groupby("ugt")["y_30"].apply(lambda s: f"{s.mean()*100:.0f}% ({int(s.sum())}/{s.notna().sum()})"))


if __name__ == "__main__":
    main()
