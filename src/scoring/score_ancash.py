"""Scoring predictivo Track A — Áncash (UGT × semana), modelo v2.

Probabilidad de protesta/bloqueo en los 30 días siguientes al cierre (domingo)
de cada semana, por UGT de ANTAMINA. Reutiliza la tabla maestra de
build_ancash (no duplica lógica de features) y respeta el anti-fuga: las
features de la semana usan solo datos hasta su domingo de cierre.

"Scoring en vivo" = semanas completas cuyo label aún no es observable
(cierre + 30 días > fecha de corte de datos). Las semanas anteriores se
reportan con predicción out-of-fold (ver src/models/train_ancash.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.dataset.build_ancash import construir
from src.models.recalibrar_umbral import cargar as cargar_umbrales
from src.models.train_ancash import MODELO_PATH, preparar_features

LOG_PRED_PATH = Path("data/processed") / "predicciones_log.csv"
LOG_REAL_PATH = Path("data/processed") / "resultados_reales.csv"


def nivel(prob: float, umbral_alto: float, umbral_medio: float) -> str:
    return "ALTO" if prob >= umbral_alto else ("MEDIO" if prob >= umbral_medio else "BAJO")


def cargar_modelo() -> dict:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(f"No se encontró {MODELO_PATH}. Corre src.models.train_ancash primero.")
    return joblib.load(MODELO_PATH)


def puntuar(master: pd.DataFrame, paquete: dict, umbrales: dict) -> pd.DataFrame:
    """Probabilidad del modelo final para TODAS las filas de `master`."""
    df = preparar_features(master)
    df["prob"] = paquete["model"].predict_proba(df[paquete["feature_cols"]])[:, 1]
    df["nivel"] = df["prob"].apply(lambda p: nivel(p, umbrales["umbral_alto"], umbrales["umbral_medio"]))
    return df


def scoring_en_vivo(master: pd.DataFrame | None = None, n_min_semanas: int = 2) -> pd.DataFrame:
    """Semanas sin label observable (mínimo las últimas `n_min_semanas`)."""
    master = construir() if master is None else master
    paquete, umbrales = cargar_modelo(), cargar_umbrales()
    df = puntuar(master, paquete, umbrales)
    semanas = sorted(df["semana_fin"].unique())
    sin_label = df.loc[df[paquete["target"]].isna(), "semana_fin"].unique()
    vivas = set(sin_label) | set(semanas[-n_min_semanas:])
    return df[df["semana_fin"].isin(vivas)].sort_values(["semana_fin", "ugt"]).reset_index(drop=True)


def predecir() -> pd.DataFrame:
    """Compatibilidad con los dashboards internos (dashboard_ancash*.py):
    última semana completa, columnas ugt / semana_scoring / probabilidad."""
    vivo = scoring_en_vivo()
    ult = vivo[vivo["semana_fin"] == vivo["semana_fin"].max()].copy()
    ult = ult.rename(columns={"semana_fin": "semana_scoring", "prob": "probabilidad"})
    cols = ["ugt", "semana_scoring", "probabilidad", "nivel", "rep_compromiso_4w", "rep_antamina_neg_4w"]
    return ult[cols].sort_values("probabilidad", ascending=False).reset_index(drop=True)


def _guardar_log(pred: pd.DataFrame) -> None:
    """Agrega la última semana a predicciones_log.csv (una fila por UGT)."""
    fecha_ej = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    nuevo = pd.DataFrame({
        "semana_scoring": pred["semana_fin"].dt.date.astype(str),
        "ugt": pred["ugt"],
        "probabilidad": pred["prob"].round(4),
        "nivel": pred["nivel"],
        "modelo": "v2",
        "fecha_ejecucion": fecha_ej,
    })
    LOG_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PRED_PATH.exists():
        existente = pd.read_csv(LOG_PRED_PATH)
        existente = existente[existente["semana_scoring"].astype(str) != nuevo["semana_scoring"].iloc[0]]
        nuevo = pd.concat([existente, nuevo], ignore_index=True)
    nuevo.to_csv(LOG_PRED_PATH, index=False)
    print(f"\n✓ Log guardado en {LOG_PRED_PATH}")


def _inicializar_log_real() -> None:
    if LOG_REAL_PATH.exists():
        return
    LOG_REAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["semana", "ugt", "hubo_protesta_real", "notas"]).to_csv(LOG_REAL_PATH, index=False)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    vivo = scoring_en_vivo()
    ultima = vivo[vivo["semana_fin"] == vivo["semana_fin"].max()].sort_values("prob", ascending=False)
    print(f"Probabilidad de protesta · 30 días desde el cierre {ultima['semana_fin'].iloc[0].date()} (por UGT):\n")
    for _, r in ultima.iterrows():
        print(f"  {r['ugt']:<18} {r['prob']*100:5.1f}%   {r['nivel']}")
    _guardar_log(ultima)
    _inicializar_log_real()


if __name__ == "__main__":
    main()
