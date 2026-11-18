"""Entrenamiento + punto de control — Track A Áncash (UGT × semana), modelo v2.

Validación walk-forward CON PURGA (src/models/evaluacion_ancash.py): para
predecir la semana de cierre T solo se usan filas cuyo label ya era
observable en T. Métricas OOF: PR-AUC (principal), ROC-AUC, Brier. NUNCA
k-fold aleatorio.

Cambios respecto a v1 (julio 2026), justificados con métricas OOF en el
README (sección "Actualización 2026-11-18"):
  1. Label sin fuga: y_30 mira (domingo de cierre, +30 días]. En v1 empezaba
     el martes de la misma semana y solapaba 6 días con `inc_prot_1w`; con ese
     label la LR v1 "lograba" PR-AUC 0.86 que no existe con el label correcto.
  2. Filas de modelado desde sep-2024 (inicio de cobertura continua de las
     fuentes de label; antes el 0 es falta de registro).
  3. Features: 8 núcleo + identidad de UGT (one-hot). Se quitan mes /
     trimestre / semana_iso (estacionalidad espuria con <2 años), tasa_pobreza
     (constante), OEFA (fuente congelada en dic-2024 → la ventana de 12 meses
     decae a 0 artificialmente), Defensoría departamental y violencia.
  4. Sin class_weight="balanced": las probabilidades quedan calibradas
     (Brier 0.232 vs 0.258) y los umbrales se interpretan como probabilidad.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.build_ancash import UGTS
from src.models import evaluacion_ancash as ev

PROCESSED_DIR = Path("data/processed")
MASTER = PROCESSED_DIR / "master_ancash_ugt.parquet"
SALIDA = PROCESSED_DIR / "ancash_modelo_resultados.csv"
OOF_PATH = PROCESSED_DIR / "oof_ancash.parquet"
MODELO_PATH = Path("models") / "modelo_v2_track_A_ancash.pkl"

TARGET = "y_30"
HORIZONTE_DIAS = 30
RANDOM_STATE = 42
UMBRAL_GO = 0.05  # el modelo debe superar al baseline por +0.05 PR-AUC OOF

DIAS_SIN_PROTESTA_PREVIA = 365  # imputación explícita: "nunca hubo protesta"

UGT_DUMMIES = ["ugt_" + u.lower().replace(" ", "_") for u in UGTS]

FEATURES_NUCLEO = [
    "inc_prot_1w", "inc_prot_4w", "racha_prot", "dias_desde_ultima_prot",
    "rep_antamina_neg_4w", "rep_compromiso_4w", "es_fecha_critica", "es_semana_electoral",
]
FEATURES = FEATURES_NUCLEO + UGT_DUMMIES          # modelo v2 (desplegado)

FEATURES_V1 = [                                    # modelo v1 (julio 2026)
    "inc_prot_1w", "inc_prot_4w", "inc_viol_1w", "inc_viol_4w",
    "delta_prot", "racha_prot", "dias_desde_ultima_prot",
    "n_feriados", "es_semana_electoral", "dias_hasta_eleccion", "es_fecha_critica",
    "mes", "trimestre", "semana_iso",
    "tasa_pobreza", "def_escalamiento_ancash", "oefa_denuncias_mineria_12m",
    "rep_antamina_neg_4w", "rep_compromiso_4w",
]
FEATURES_RF = [f for f in FEATURES_V1 if f not in (
    "mes", "trimestre", "semana_iso", "tasa_pobreza", "oefa_denuncias_mineria_12m")] + UGT_DUMMIES


def _lr(balanced: bool) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced" if balanced else None,
                                   max_iter=2000, random_state=RANDOM_STATE)),
    ])


# nombre → (pipeline, features, descripción)
CANDIDATOS = {
    "v1_lr_19f_balanced": (_lr(True), FEATURES_V1, "Especificación v1 (julio 2026) reevaluada sin fuga"),
    "v2_lr_nucleo_ugt": (_lr(False), FEATURES, "LR sin balanceo, 8 features núcleo + UGT"),
    "rf_15f_ugt": (Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                       class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ]), FEATURES_RF, "Random forest de contraste"),
}
MODELO_FINAL = "v2_lr_nucleo_ugt"
MODELO_ANTERIOR = "v1_lr_19f_balanced"


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derivaciones comunes a entrenamiento y scoring (no usa el futuro)."""
    df = df.copy()
    for u, col in zip(UGTS, UGT_DUMMIES):
        df[col] = (df["ugt"] == u).astype(int)
    df["dias_desde_ultima_prot"] = df["dias_desde_ultima_prot"].fillna(DIAS_SIN_PROTESTA_PREVIA)
    return df


def entrenar(master: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """Evalúa candidatos (OOF), aplica go/no-go, guarda OOF del modelo final y
    el modelo final entrenado con todas las filas con label observable."""
    log = print if verbose else (lambda *a, **k: None)
    master = pd.read_parquet(MASTER) if master is None else master
    master = preparar_features(master)
    df = ev.filas_modelables(master, TARGET)
    y = df[TARGET].astype(int)
    log(f"Filas modelables: {len(df)} ({df['semana_fin'].min().date()} → {df['semana_fin'].max().date()}), "
        f"{y.mean()*100:.1f}% positivos ({TARGET})")

    oofs = {n: ev.walk_forward_oof(p, df, f, TARGET, HORIZONTE_DIAS) for n, (p, f, _) in CANDIDATOS.items()}
    base = ev.baseline_tasa_ugt(df, TARGET, HORIZONTE_DIAS)
    comun = pd.concat([base, *oofs.values()], axis=1).notna().all(axis=1)

    filas = [{"tipo": "baseline", "nombre": "tasa_historica_ugt", **ev.metricas(y[comun], base[comun])}]
    for n, p in oofs.items():
        filas.append({"tipo": "modelo", "nombre": n, **ev.metricas(y[comun], p[comun]),
                      "descripcion": CANDIDATOS[n][2]})
    res = pd.DataFrame(filas)
    res.to_csv(SALIDA, index=False)
    log(f"\nOOF común: {int(comun.sum())} filas desde {df.loc[comun, 'semana_fin'].min().date()}")
    log(res[["nombre", "n", "pr_auc", "roc_auc", "brier"]].to_string(index=False))

    m_final = res.set_index("nombre").loc[MODELO_FINAL]
    m_base = res.set_index("nombre").loc["tasa_historica_ugt"]
    dif = float(m_final["pr_auc"] - m_base["pr_auc"])
    es_go = dif >= UMBRAL_GO
    log(f"\nPUNTO DE CONTROL ({TARGET}): {MODELO_FINAL} PR-AUC {m_final['pr_auc']:.4f} vs baseline "
        f"{m_base['pr_auc']:.4f} → Δ {dif:+.4f} (umbral +{UMBRAL_GO}) → {'GO' if es_go else 'NO-GO'}")
    if not es_go:
        raise SystemExit("NO-GO: el modelo no supera al baseline. No se guarda modelo nuevo (CLAUDE.md).")

    # OOF del modelo final (todas las filas donde hubo modelo previo posible)
    oof = df[["ugt", "semana_inicio", "semana_fin", TARGET]].copy()
    oof["prob"] = oofs[MODELO_FINAL]
    oof = oof[oof["prob"].notna()].reset_index(drop=True)
    oof.to_parquet(OOF_PATH, index=False)

    pipe, feats, _ = CANDIDATOS[MODELO_FINAL]
    modelo = clone(pipe).fit(df[feats], y)
    paquete = {
        "model": modelo,
        "feature_cols": feats,
        "modelo_tipo": "logistic_regression",
        "nombre_candidato": MODELO_FINAL,
        "target": TARGET,
        "horizonte_dias": HORIZONTE_DIAS,
        "metricas_oof": {k: float(m_final[k]) for k in ("pr_auc", "roc_auc", "brier")},
        "metricas_oof_baseline": {k: float(m_base[k]) for k in ("pr_auc", "roc_auc", "brier")},
        # claves de v1 que leen los dashboards internos (dashboard_ancash*.py)
        "pr_auc_cv": float(m_final["pr_auc"]),
        "pr_auc_baseline": float(m_base["pr_auc"]),
        "n_entrenamiento": int(len(df)),
        "entrenado_hasta": str(df["semana_fin"].max().date()),  # última semana con label observable
        "fecha_corte_datos": str(pd.Timestamp(master["fecha_corte_datos"].iloc[0]).date()),
        "trained_on": str(pd.Timestamp.now().date()),
    }
    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, MODELO_PATH)
    log(f"✓ Modelo final guardado en {MODELO_PATH} ({len(df)} filas, label observable hasta {paquete['entrenado_hasta']})")
    log(f"✓ OOF guardado en {OOF_PATH} ({len(oof)} filas)")
    return {"paquete": paquete, "resultados": res, "oof": oof}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    entrenar()


if __name__ == "__main__":
    main()
