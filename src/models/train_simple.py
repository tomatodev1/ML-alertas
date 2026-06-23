import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MASTER = PROCESSED_DIR / "master_zona_semana.parquet"
SALIDA = PROCESSED_DIR / "simple_model_results.csv"

# ── Configuración ─────────────────────────────────────────────────────────────
N_SPLITS = 5          # walk-forward con 5 folds
TARGET = "y_30"       # empezar con horizonte 30 días
RANDOM_STATE = 42
PRECISION_OBJETIVO = 0.5

# zona/semana_inicio/track son identificadores; semana_iso y mes_nombre son
# redundantes con año/mes/trimestre ya presentes; y_* son las etiquetas.
EXCLUDE_COLS = [
    "zona", "semana_inicio", "semana_iso", "track", "mes_nombre",
    "y_7", "y_14", "y_30", "y_60",
]

MODELOS = {
    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
}


# ── Métricas ──────────────────────────────────────────────────────────────────
def _recall_a_precision(y_true: np.ndarray, y_score: np.ndarray, precision_obj: float = PRECISION_OBJETIVO) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= precision_obj
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


# ── Preparación de datos ──────────────────────────────────────────────────────
def _preparar_track(master: pd.DataFrame, track: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = master[(master["track"] == track) & master[target].notna()].copy()
    # Orden temporal estricto: TimeSeriesSplit asume que el índice de fila = orden temporal
    df = df.sort_values(["semana_inicio", "zona"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df[target].astype(int)
    return X, y


# ── Evaluación walk-forward ───────────────────────────────────────────────────
def evaluar_modelo(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> tuple[list[float], list[float]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pr_aucs: list[float] = []
    recalls: list[float] = []

    for idx_train, idx_test in tscv.split(X):
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue  # sin 2 clases no se puede entrenar o medir PR-AUC en el fold

        modelo = clone(pipeline)
        modelo.fit(X_train, y_train)
        y_score = modelo.predict_proba(X_test)[:, 1]

        pr_aucs.append(average_precision_score(y_test, y_score))
        recalls.append(_recall_a_precision(y_test, y_score))

    return pr_aucs, recalls


# ── Reporte ───────────────────────────────────────────────────────────────────
def _imprimir_tabla(df: pd.DataFrame) -> None:
    print(f"\n{'Modelo':<20} | {'Track':<5} | {'PR-AUC medio':<13} | {'PR-AUC std':<11} | {'Recall@P50':<10}")
    print("-" * 75)
    for _, fila in df.iterrows():
        print(
            f"{fila['modelo']:<20} | {fila['track']:<5} | {fila['pr_auc_medio']:<13} | "
            f"±{fila['pr_auc_std']:<10} | {fila['recall_p50_medio']:<10}"
        )


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)

    resultados: list[dict] = []
    for track in ["A", "B"]:
        X, y = _preparar_track(master, track, TARGET)

        for nombre_modelo, pipeline in MODELOS.items():
            pr_aucs, recalls = evaluar_modelo(pipeline, X, y)

            resultados.append({
                "modelo": nombre_modelo,
                "track": track,
                "horizonte": TARGET,
                "pr_auc_medio": round(float(np.mean(pr_aucs)), 4) if pr_aucs else np.nan,
                "pr_auc_std": round(float(np.std(pr_aucs)), 4) if pr_aucs else np.nan,
                "recall_p50_medio": round(float(np.mean(recalls)), 4) if recalls else np.nan,
                "n_folds_validos": len(pr_aucs),
            })

    df_resultados = pd.DataFrame(resultados)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(SALIDA, index=False)

    _imprimir_tabla(df_resultados)
    print(f"\n✓ Resultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()
