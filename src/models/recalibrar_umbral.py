import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.models.train_xgboost import FEATURE_COLS_B, _modelo_lgb, _modelo_xgb, _preparar, _scale_pos_weight

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MASTER = PROCESSED_DIR / "master_zona_semana.parquet"
MODELO_PKL = MODELS_DIR / "modelo_v1_track_B.pkl"
SALIDA_CALIBRACION = PROCESSED_DIR / "calibracion_umbral.csv"

# ── Configuración ─────────────────────────────────────────────────────────────
TARGET = "y_30"
N_SPLITS = 5
FOLDS_CALIBRACION = [1, 2, 3]  # folds 2, 3, 4 (0-indexed) — excluye fold 1 (poco dato) y 5 (sesgo electoral)
UMBRALES = np.round(np.arange(0.20, 0.71, 0.05), 2)
RECALL_MIN = 0.40
PRECISION_MIN = 0.35


# ── Scores por fold ────────────────────────────────────────────────────────────
def _scores_por_fold(
    X: pd.DataFrame, y: pd.Series, folds_idx: list[int], modelo_tipo: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Entrena el mismo tipo de modelo guardado (xgboost o lightgbm) walk-forward
    y devuelve (y_test, score) para cada fold solicitado."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    todos_los_folds = list(tscv.split(X))
    resultados = []
    for i in folds_idx:
        idx_train, idx_test = todos_los_folds[i]
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        if modelo_tipo == "xgboost":
            modelo = _modelo_xgb(_scale_pos_weight(y_train))
            modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            modelo = _modelo_lgb()
            modelo.fit(X_train, y_train)
        score = modelo.predict_proba(X_test)[:, 1]
        resultados.append((y_test.values, score))
    return resultados


# ── Evaluación de umbrales ──────────────────────────────────────────────────────
def evaluar_umbrales(resultados_folds: list[tuple[np.ndarray, np.ndarray]], umbrales: np.ndarray) -> pd.DataFrame:
    filas = []
    for umbral in umbrales:
        precisions, recalls, f1s, tasas_alerta = [], [], [], []
        for y_test, score in resultados_folds:
            y_pred = (score >= umbral).astype(int)
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            tasas_alerta.append(y_pred.mean())
        filas.append({
            "umbral": round(float(umbral), 2),
            "precision_media": round(float(np.mean(precisions)), 4),
            "recall_media": round(float(np.mean(recalls)), 4),
            "f1_media": round(float(np.mean(f1s)), 4),
            "tasa_alerta_media": round(float(np.mean(tasas_alerta)), 4),
        })
    return pd.DataFrame(filas)


def elegir_umbral(tabla: pd.DataFrame) -> tuple[pd.Series, bool]:
    cumple = tabla[(tabla["recall_media"] >= RECALL_MIN) & (tabla["precision_media"] >= PRECISION_MIN)]
    if not cumple.empty:
        return cumple.loc[cumple["f1_media"].idxmax()], True
    return tabla.loc[tabla["f1_media"].idxmax()], False


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)
    X, y = _preparar(master, "B", TARGET, FEATURE_COLS_B)

    modelo_tipo = joblib.load(MODELO_PKL)["modelo_tipo"]
    print(f"Recalibrando umbral para modelo guardado: {modelo_tipo}")

    resultados_folds = _scores_por_fold(X, y, FOLDS_CALIBRACION, modelo_tipo)
    tabla = evaluar_umbrales(resultados_folds, UMBRALES)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA_CALIBRACION, index=False)

    elegido, cumple_ambas = elegir_umbral(tabla)

    print(f"\n{'Umbral':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<8} | {'Tasa alerta':<12}")
    print("-" * 58)
    for _, fila in tabla.iterrows():
        marca = "  <-- elegido" if fila["umbral"] == elegido["umbral"] else ""
        print(
            f"{fila['umbral']:<8} | {fila['precision_media']:<10} | {fila['recall_media']:<8} | "
            f"{fila['f1_media']:<8} | {fila['tasa_alerta_media'] * 100:<10.1f}%{marca}"
        )

    print(f"""
Umbral recalibrado: {elegido['umbral']}
Precision promedio (folds 2-4): {elegido['precision_media']}
Recall promedio (folds 2-4):    {elegido['recall_media']}
F1 promedio (folds 2-4):        {elegido['f1_media']}
Tasa de alerta esperada:        {elegido['tasa_alerta_media'] * 100:.1f}% de semanas dispararían alerta
Cumple ambos pisos (recall>={RECALL_MIN}, precision>={PRECISION_MIN}): {"sí" if cumple_ambas else "NO -- se usó el de mejor F1 como fallback"}
""")

    # ── Actualizar el modelo guardado con el nuevo umbral ────────────────────
    paquete = joblib.load(MODELO_PKL)
    umbral_anterior = paquete["threshold"]
    paquete["threshold"] = float(elegido["umbral"])
    paquete["threshold_calibracion"] = {
        "metodo": "folds 2-4 de TimeSeriesSplit(5); excluye fold 1 (poco dato) y fold 5 (sesgo electoral 2026)",
        "precision_media": float(elegido["precision_media"]),
        "recall_media": float(elegido["recall_media"]),
        "f1_media": float(elegido["f1_media"]),
        "tasa_alerta_media": float(elegido["tasa_alerta_media"]),
        "cumple_pisos_recall_precision": bool(cumple_ambas),
        "umbral_anterior": umbral_anterior,
    }
    joblib.dump(paquete, MODELO_PKL)

    print(f"✓ Modelo actualizado en {MODELO_PKL} (umbral {umbral_anterior} -> {elegido['umbral']})")
    print(f"✓ Calibración guardada en {SALIDA_CALIBRACION}")


if __name__ == "__main__":
    main()
