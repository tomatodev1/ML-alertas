import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.dataset.build_master import FECHA_CORTE, _cargar_defensoria

warnings.filterwarnings("ignore", category=UserWarning)

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MASTER = PROCESSED_DIR / "master_zona_semana.parquet"
BASELINE_CSV = PROCESSED_DIR / "baseline_results.csv"

# ── Configuración ─────────────────────────────────────────────────────────────
HORIZONTES = [7, 14, 30, 60]
TARGET_PRINCIPAL = "y_30"
N_SPLITS = 5
RANDOM_STATE = 42
RECALL_MIN_UMBRAL = 0.4

# Features de Track B — se excluyen las de calendario puro que no aportan señal
# (mes/trimestre/semana_iso sí se incluyen: capturan estacionalidad de paros agrícolas).
# Nombres de columnas mapeados a los reales de master_zona_semana.parquet:
#   es_feriado → n_feriados, dias_a_eleccion_mas_cercana → dias_hasta_eleccion,
#   semana_del_año → semana_iso.
FEATURE_COLS_B = [
    "n_crit_1w", "n_alrt_1w", "n_info_1w", "n_total_1w",
    "n_protesta_1w", "n_violencia_1w", "n_politica_1w",
    "n_crit_4w", "n_protesta_4w",
    "delta_crit", "delta_protesta",
    "dias_desde_ultima_crit", "racha_semanas_con_alerta",
    "def_escalamiento_global",
    "tasa_pobreza",
    "n_feriados", "es_semana_electoral", "dias_hasta_eleccion",
    "es_fecha_critica", "mes", "trimestre", "semana_iso",
    # BD_Incidentes interna (geolocalizada, resolución semanal real)
    "inc_n_protesta_1w", "inc_n_violencia_1w", "inc_n_protesta_4w",
]
# Track A: mismo set + escalamiento propio de zona (las 5 zonas de Track A
# reportan directo a Defensoría, a diferencia de las sub-zonas de Track B).
FEATURE_COLS_A = FEATURE_COLS_B + ["def_escalamiento_zona"]


# ── Preparación de datos ──────────────────────────────────────────────────────
def _preparar(master: pd.DataFrame, track: str, target: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = master[(master["track"] == track) & master[target].notna()].copy()
    df = df.sort_values(["semana_inicio", "zona"]).reset_index(drop=True)
    X = df[feature_cols].fillna(0)  # xgb/lgb toleran NaN, pero 0 es más interpretable aquí
    y = df[target].astype(int)
    return X, y


# ── Métricas ──────────────────────────────────────────────────────────────────
def _recall_a_precision(y_true: np.ndarray, y_score: np.ndarray, precision_obj: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= precision_obj
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def _scale_pos_weight(y_train: pd.Series) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    return neg / pos if pos > 0 else 1.0


# ── Modelos ───────────────────────────────────────────────────────────────────
def _modelo_xgb(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        importance_type="gain",  # comparable a la importancia gini de random forest
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _modelo_lgb() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        metric="average_precision",
        importance_type="gain",  # default es "split" (conteo); "gain" es comparable a RF
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


# ── Validación walk-forward (XGB + LGB en los mismos folds) ──────────────────
def evaluar_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> tuple[dict, dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    resultados = {
        "xgboost": {"pr_auc": [], "recall_p4": [], "recall_p5": []},
        "lightgbm": {"pr_auc": [], "recall_p4": [], "recall_p5": []},
    }
    ultimo_fold: dict = {}

    for idx_train, idx_test in tscv.split(X):
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        spw = _scale_pos_weight(y_train)
        modelo_x = _modelo_xgb(spw)
        modelo_x.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        score_x = modelo_x.predict_proba(X_test)[:, 1]

        modelo_l = _modelo_lgb()
        modelo_l.fit(X_train, y_train)
        score_l = modelo_l.predict_proba(X_test)[:, 1]

        resultados["xgboost"]["pr_auc"].append(average_precision_score(y_test, score_x))
        resultados["xgboost"]["recall_p4"].append(_recall_a_precision(y_test, score_x, 0.4))
        resultados["xgboost"]["recall_p5"].append(_recall_a_precision(y_test, score_x, 0.5))

        resultados["lightgbm"]["pr_auc"].append(average_precision_score(y_test, score_l))
        resultados["lightgbm"]["recall_p4"].append(_recall_a_precision(y_test, score_l, 0.4))
        resultados["lightgbm"]["recall_p5"].append(_recall_a_precision(y_test, score_l, 0.5))

        ultimo_fold = {
            "y_test": y_test.values,
            "score_xgb": score_x,
            "score_lgb": score_l,
            "modelo_xgb": modelo_x,
            "modelo_lgb": modelo_l,
        }

    return resultados, ultimo_fold


def _media(lista: list[float]) -> float:
    return float(np.mean(lista)) if lista else float("nan")


# ── Paso 4: búsqueda de umbral óptimo ─────────────────────────────────────────
def buscar_umbral_optimo(y_true: np.ndarray, y_score: np.ndarray, recall_min: float = RECALL_MIN_UMBRAL) -> dict:
    mejor = None
    for umbral in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_score >= umbral).astype(int)
        if y_pred.sum() == 0:
            continue
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall < recall_min:
            continue
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if mejor is None or f1 > mejor["f1"]:
            mejor = {"umbral": round(float(umbral), 2), "precision": round(float(precision), 4),
                     "recall": round(float(recall), 4), "f1": round(float(f1), 4)}

    if mejor is None:
        # Ningún umbral alcanza recall_min: relajar y reportar el de mejor F1 puro
        for umbral in np.arange(0.1, 0.91, 0.01):
            y_pred = (y_score >= umbral).astype(int)
            if y_pred.sum() == 0:
                continue
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if mejor is None or f1 > mejor["f1"]:
                mejor = {"umbral": round(float(umbral), 2), "precision": round(float(precision), 4),
                         "recall": round(float(recall), 4), "f1": round(float(f1), 4)}
    return mejor or {"umbral": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}


# ── Paso 6: label relajado de Track A ─────────────────────────────────────────
def _label_track_a_relajado(master: pd.DataFrame, defensoria: pd.DataFrame, h: int) -> pd.Series:
    """y_h_relajado = 1 si el reporte de Defensoría del mes que cubre semana_inicio+h
    muestra escalamiento_zona >= 1 (sin exigir que sea NUEVO respecto al mes anterior)."""
    df_a = master[master["track"] == "A"]
    fecha_futura = df_a["semana_inicio"] + pd.Timedelta(days=h)
    tmp = df_a[["zona", "semana_inicio"]].copy()
    tmp["_año_lab"] = fecha_futura.dt.year
    tmp["_mes_lab"] = fecha_futura.dt.month

    def_esc = defensoria.rename(columns={"año": "_año_lab", "mes_num": "_mes_lab"})[
        ["zona", "_año_lab", "_mes_lab", "escalamiento_zona", "escalamiento_global"]
    ]
    merged = tmp.merge(def_esc, on=["zona", "_año_lab", "_mes_lab"], how="left")
    merged.index = df_a.index

    label = (merged["escalamiento_zona"].fillna(0) >= 1).astype(float)
    label[merged["escalamiento_global"] == 0] = np.nan  # mes aún no publicado por Defensoría
    label[(fecha_futura > FECHA_CORTE).values] = np.nan
    return label


# ── Reportes ──────────────────────────────────────────────────────────────────
def _tabla_multihorizonte(master: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for h in HORIZONTES:
        target = f"y_{h}"
        X, y = _preparar(master, "B", target, FEATURE_COLS_B)
        resultados, _ = evaluar_cv(X, y)

        base_sub = baseline_df[
            (baseline_df["track"] == "B")
            & (baseline_df["horizonte"] == target)
            & (baseline_df["baseline"] == "tasa_historica")
        ]
        pr_auc_base = float(base_sub["pr_auc"].iloc[0]) if not base_sub.empty else float("nan")

        filas.append({
            "horizonte": target,
            "pr_auc_xgb": round(_media(resultados["xgboost"]["pr_auc"]), 4),
            "pr_auc_lgb": round(_media(resultados["lightgbm"]["pr_auc"]), 4),
            "pr_auc_baseline_tasa_historica": round(pr_auc_base, 4),
        })
    return pd.DataFrame(filas)


def _imprimir_tabla_horizontes(df: pd.DataFrame) -> None:
    print(f"\n{'Horizonte':<10} | {'PR-AUC XGB':<11} | {'PR-AUC LGB':<11} | {'Baseline tasa_historica':<23}")
    print("-" * 65)
    for _, fila in df.iterrows():
        print(
            f"{fila['horizonte']:<10} | {fila['pr_auc_xgb']:<11} | {fila['pr_auc_lgb']:<11} | "
            f"{fila['pr_auc_baseline_tasa_historica']:<23}"
        )


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)
    baseline_df = pd.read_csv(BASELINE_CSV)
    defensoria = _cargar_defensoria()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Paso 5: multi-horizonte (incluye Paso 1-3 para cada horizonte) ────────
    print("Evaluando Track B multi-horizonte (walk-forward, 5 folds)...")
    tabla_h = _tabla_multihorizonte(master, baseline_df)
    tabla_h.to_csv(PROCESSED_DIR / "xgb_lgb_multihorizonte_track_b.csv", index=False)
    _imprimir_tabla_horizontes(tabla_h)

    # ── Paso 2-4 en detalle para y_30 (umbral óptimo) ──────────────────────────
    print(f"\nDetalle {TARGET_PRINCIPAL} Track B (para selección de umbral)...")
    X_b, y_b = _preparar(master, "B", TARGET_PRINCIPAL, FEATURE_COLS_B)
    resultados_b, ultimo_fold_b = evaluar_cv(X_b, y_b)

    pr_auc_xgb = _media(resultados_b["xgboost"]["pr_auc"])
    pr_auc_lgb = _media(resultados_b["lightgbm"]["pr_auc"])
    ganador = "xgboost" if pr_auc_xgb >= pr_auc_lgb else "lightgbm"
    print(f"  PR-AUC medio XGBoost:  {pr_auc_xgb:.4f}")
    print(f"  PR-AUC medio LightGBM: {pr_auc_lgb:.4f}")
    print(f"  Modelo ganador: {ganador}")

    score_ganador = ultimo_fold_b["score_xgb"] if ganador == "xgboost" else ultimo_fold_b["score_lgb"]
    umbral_info = buscar_umbral_optimo(ultimo_fold_b["y_test"], score_ganador)
    print(
        f"\n  Umbral óptimo (último fold, recall>={RECALL_MIN_UMBRAL}): {umbral_info['umbral']}\n"
        f"  Precision: {umbral_info['precision']} | Recall: {umbral_info['recall']} | F1: {umbral_info['f1']}"
    )

    # ── Paso 6: Track A con label relajado ─────────────────────────────────────
    print(f"\nTrack A con label relajado (escalamiento_zona >= 1, {TARGET_PRINCIPAL})...")
    label_relajado = _label_track_a_relajado(master, defensoria, 30)
    df_a = master[master["track"] == "A"].copy()
    df_a["y_30_relajado"] = label_relajado
    df_a_validas = df_a[df_a["y_30_relajado"].notna()].sort_values(["semana_inicio", "zona"]).reset_index(drop=True)

    balance_relajado = float(df_a_validas["y_30_relajado"].mean())
    print(f"  Balance y_30 original (Track A): 5.6% positivos (referencia Fase 2)")
    print(f"  Balance y_30_relajado (Track A): {balance_relajado * 100:.1f}% positivos ({len(df_a_validas)} filas)")

    X_a = df_a_validas[FEATURE_COLS_A].fillna(0)
    y_a = df_a_validas["y_30_relajado"].astype(int)
    resultados_a, _ = evaluar_cv(X_a, y_a)
    pr_auc_a_xgb = _media(resultados_a["xgboost"]["pr_auc"])
    pr_auc_a_lgb = _media(resultados_a["lightgbm"]["pr_auc"])
    print(f"  PR-AUC medio XGBoost (label relajado):  {pr_auc_a_xgb:.4f}")
    print(f"  PR-AUC medio LightGBM (label relajado): {pr_auc_a_lgb:.4f}")

    base_a_sub = baseline_df[
        (baseline_df["track"] == "A") & (baseline_df["horizonte"] == "y_30") & (baseline_df["baseline"] == "tasa_historica")
    ]
    pr_auc_base_a = float(base_a_sub["pr_auc"].iloc[0]) if not base_a_sub.empty else float("nan")
    print(f"  Referencia baseline tasa_historica (label original y_30): {pr_auc_base_a:.4f}")

    # ── Paso 7: guardar el mejor modelo (Track B, y_30) sobre TODOS los datos ──
    print(f"\nReentrenando {ganador} sobre todos los datos disponibles de Track B...")
    if ganador == "xgboost":
        spw_full = _scale_pos_weight(y_b)
        modelo_final = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, importance_type="gain",
            scale_pos_weight=spw_full, random_state=RANDOM_STATE, n_jobs=-1,
        )
        modelo_final.fit(X_b, y_b)
    else:
        modelo_final = _modelo_lgb()
        modelo_final.fit(X_b, y_b)

    ruta_modelo = MODELS_DIR / "modelo_v1_track_B.pkl"
    joblib.dump({
        "model": modelo_final,
        "threshold": umbral_info["umbral"],
        "feature_cols": FEATURE_COLS_B,
        "track": "B",
        "target": TARGET_PRINCIPAL,
        "pr_auc_cv": pr_auc_xgb if ganador == "xgboost" else pr_auc_lgb,
        "modelo_tipo": ganador,
        "trained_on": str(pd.Timestamp.now().date()),
    }, ruta_modelo)
    print(f"  ✓ Modelo guardado en {ruta_modelo}")

    importancias_finales = pd.DataFrame({
        "feature": FEATURE_COLS_B,
        "importance": modelo_final.feature_importances_,
        "modelo": ganador,
    }).sort_values("importance", ascending=False)
    ruta_importancia = PROCESSED_DIR / "feature_importance_xgb.csv"
    importancias_finales.to_csv(ruta_importancia, index=False)
    print(f"  ✓ Feature importance guardada en {ruta_importancia}")

    print("\nTop 10 features del modelo final (Track B):")
    print(importancias_finales.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
