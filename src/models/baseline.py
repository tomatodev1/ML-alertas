import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MASTER = PROCESSED_DIR / "master_zona_semana.parquet"
SALIDA = PROCESSED_DIR / "baseline_results.csv"

HORIZONTES = [7, 14, 30, 60]
N_SEMANAS_TRAIN = 80  # primeras 80 semanas para entrenar, resto para validar
PRECISION_OBJETIVO = 0.5


# ── Split temporal ────────────────────────────────────────────────────────────
def _split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primeras N_SEMANAS_TRAIN semanas → train, el resto → val. NUNCA aleatorio."""
    semanas = sorted(df["semana_inicio"].unique())
    semanas_train = set(semanas[:N_SEMANAS_TRAIN])
    train = df[df["semana_inicio"].isin(semanas_train)].copy()
    val = df[~df["semana_inicio"].isin(semanas_train)].copy()
    return train, val


# ── Métricas ──────────────────────────────────────────────────────────────────
def _recall_a_precision(y_true: np.ndarray, y_score: np.ndarray, precision_obj: float) -> float:
    """Recall máximo entre los puntos de la curva PR con precisión >= precision_obj."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= precision_obj
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def _f1_en_umbral(y_true: np.ndarray, y_score: np.ndarray, umbral: float = 0.5) -> float:
    y_pred = (y_score >= umbral).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


# ── Baseline "semana anterior" ────────────────────────────────────────────────
def _semana_anterior_pred(df_track: pd.DataFrame, col_y: str) -> pd.Series:
    """Predicción para la semana t = valor real de y en la semana t-1, por zona."""
    ordered = df_track.sort_values(["zona", "semana_inicio"])
    pred = ordered.groupby("zona")[col_y].shift(1)
    return pred.reindex(df_track.index)


# ── Evaluación principal ──────────────────────────────────────────────────────
def evaluar_baselines(master: pd.DataFrame) -> pd.DataFrame:
    resultados: list[dict] = []

    for track in ["A", "B"]:
        df_track = master[master["track"] == track].copy()

        for h in HORIZONTES:
            col_y = f"y_{h}"
            sub = df_track[df_track[col_y].notna()].copy()
            sub = sub.sort_values(["zona", "semana_inicio"])

            train, val = _split_temporal(sub)
            if val.empty or val[col_y].nunique() < 2:
                print(f"  AVISO: track={track} {col_y} sin suficiente variación en validación, se omite.")
                continue

            y_train = train[col_y].values
            y_val = val[col_y].values

            pred_prev_full = _semana_anterior_pred(sub, col_y)
            pred_prev_val = pred_prev_full.loc[val.index].fillna(0).values

            baselines_pred = {
                "siempre_negativo": np.zeros(len(val)),
                "tasa_historica":   np.full(len(val), y_train.mean()),
                "semana_anterior":  pred_prev_val,
            }

            for nombre, y_score in baselines_pred.items():
                pr_auc = average_precision_score(y_val, y_score)
                recall_p50 = _recall_a_precision(y_val, y_score, PRECISION_OBJETIVO)
                f1 = _f1_en_umbral(y_val, y_score)

                resultados.append({
                    "baseline": nombre,
                    "horizonte": col_y,
                    "track": track,
                    "pr_auc": round(pr_auc, 4),
                    "recall_p50": round(recall_p50, 4),
                    "f1": round(f1, 4),
                    "n_val": len(val),
                    "tasa_positivos_val": round(float(y_val.mean()), 4),
                })

    return pd.DataFrame(resultados)


# ── Reporte ───────────────────────────────────────────────────────────────────
def _imprimir_tabla(df: pd.DataFrame) -> None:
    print(f"\n{'Baseline':<18} | {'Horizonte':<9} | {'Track':<5} | {'PR-AUC':<7} | {'Recall@P50':<10} | {'F1':<6}")
    print("-" * 70)
    for _, fila in df.iterrows():
        print(
            f"{fila['baseline']:<18} | {fila['horizonte']:<9} | {fila['track']:<5} | "
            f"{fila['pr_auc']:<7} | {fila['recall_p50']:<10} | {fila['f1']:<6}"
        )


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)
    resultados = evaluar_baselines(master)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(SALIDA, index=False)

    _imprimir_tabla(resultados)
    print(f"\n✓ Resultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()
