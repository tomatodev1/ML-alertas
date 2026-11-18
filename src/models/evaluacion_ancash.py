"""Evaluación walk-forward con purga — Track A Áncash (UGT × semana).

Simula exactamente lo que pasa en producción: para predecir la semana de
cierre T, el modelo se entrena SOLO con filas cuyo label ya era observable en
T, es decir semana_fin + HORIZONTE <= T. Sin esta purga, las últimas filas de
entrenamiento tienen un label que mira hacia el periodo de prueba (fuga).

Se reentrena cada `PASO_SEMANAS` semanas (bloques de prueba consecutivos) con
ventana expansiva. NUNCA k-fold aleatorio.

Las predicciones out-of-fold (OOF) resultantes se usan para:
  - comparar modelos (PR-AUC, ROC-AUC, Brier),
  - calibrar los umbrales ALTO/MEDIO,
  - poblar la serie histórica del dashboard (origen = oof_backfill).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Inicio de cobertura CONTINUA de las fuentes de label (dataIncidentes +
# dataFormulario). Antes de sep-2024 hay 0–8 eventos/mes en todo Áncash: el
# label 0 de esas semanas es falta de registro, no ausencia de protesta.
INICIO_MODELADO = pd.Timestamp("2024-09-08")  # primer domingo de cierre de sep-2024

PASO_SEMANAS = 4          # reentrenar cada 4 semanas
MIN_SEMANAS_TRAIN = 16    # mínimo de semanas con label observable para el 1er modelo


def filas_modelables(master: pd.DataFrame, target: str) -> pd.DataFrame:
    df = master[(master["semana_fin"] >= INICIO_MODELADO) & master[target].notna()].copy()
    return df.sort_values(["semana_fin", "ugt"]).reset_index(drop=True)


def walk_forward_oof(pipeline, df: pd.DataFrame, features: list[str], target: str,
                     horizonte_dias: int) -> pd.Series:
    """Probabilidad OOF por fila (NaN donde no hubo modelo previo posible)."""
    semanas = np.sort(df["semana_fin"].unique())
    oof = pd.Series(np.nan, index=df.index)
    horizonte = pd.Timedelta(days=horizonte_dias)
    for i in range(0, len(semanas), PASO_SEMANAS):
        bloque = semanas[i:i + PASO_SEMANAS]
        t0 = pd.Timestamp(bloque[0])
        # label observable al momento de pronosticar t0
        tr = df["semana_fin"] + horizonte <= t0
        if df.loc[tr, "semana_fin"].nunique() < MIN_SEMANAS_TRAIN or df.loc[tr, target].nunique() < 2:
            continue
        te = df["semana_fin"].isin(bloque)
        m = clone(pipeline).fit(df.loc[tr, features], df.loc[tr, target].astype(int))
        oof.loc[te] = m.predict_proba(df.loc[te, features])[:, 1]
    return oof


def baseline_tasa_ugt(df: pd.DataFrame, target: str, horizonte_dias: int) -> pd.Series:
    """Baseline trivial: tasa histórica de la UGT con la misma purga."""
    out = pd.Series(np.nan, index=df.index)
    horizonte = pd.Timedelta(days=horizonte_dias)
    for t in np.sort(df["semana_fin"].unique()):
        t = pd.Timestamp(t)
        hist = df[df["semana_fin"] + horizonte <= t]
        if hist["semana_fin"].nunique() < MIN_SEMANAS_TRAIN:
            continue
        tasas = hist.groupby("ugt")[target].mean()
        sel = df["semana_fin"] == t
        out.loc[sel] = df.loc[sel, "ugt"].map(tasas).fillna(hist[target].mean()).values
    return out


def metricas(y: pd.Series, p: pd.Series) -> dict:
    ok = p.notna()
    y, p = y[ok].astype(int), p[ok]
    if y.nunique() < 2:
        return {"n": int(len(y)), "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan}
    return {
        "n": int(len(y)),
        "tasa_positivos": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def metricas_por_nivel(y: pd.Series, p: pd.Series, umbral_alto: float, umbral_medio: float) -> pd.DataFrame:
    """Para cada nivel: n de semanas-UGT, tasa de acierto (precisión: % de
    semanas del nivel que efectivamente tuvieron protesta en los 30 días
    siguientes — para BAJO, % que NO tuvieron) y recall (fracción de los
    positivos — o de los negativos, para BAJO — que cayeron en el nivel)."""
    ok = p.notna()
    y, p = y[ok].astype(int), p[ok]
    nivel = np.where(p >= umbral_alto, "ALTO", np.where(p >= umbral_medio, "MEDIO", "BAJO"))
    filas = []
    for nv, umbral in [("ALTO", umbral_alto), ("MEDIO", umbral_medio), ("BAJO", 0.0)]:
        sel = nivel == nv
        n = int(sel.sum())
        if nv == "BAJO":
            acierto = float((y[sel] == 0).mean()) if n else np.nan
            recall = float(sel[y.values == 0].mean()) if (y == 0).any() else np.nan
        else:
            acierto = float(y[sel].mean()) if n else np.nan
            recall = float(sel[y.values == 1].mean()) if (y == 1).any() else np.nan
        filas.append({"nivel": nv, "umbral": umbral, "n": n,
                      "tasa_acierto": round(acierto, 4) if n else np.nan,
                      "recall": round(recall, 4)})
    return pd.DataFrame(filas)


def precision_recall_acumulado(y: pd.Series, p: pd.Series, umbral: float) -> tuple[float, float, float]:
    """Precisión/recall de 'alerta si p >= umbral' (acumulado, no por banda)."""
    ok = p.notna()
    y, p = y[ok].astype(int).values, p[ok].values
    pred = p >= umbral
    prec = float(y[pred].mean()) if pred.any() else np.nan
    rec = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    return prec, rec, float(pred.mean())
