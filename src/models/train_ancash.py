"""Entrenamiento + punto de control Track A regional (Áncash, UGT × semana).

Valida si, con la BD de incidentes como fuente de label semanal, un modelo
supera al baseline trivial (go/no-go). Validación temporal walk-forward,
métricas PR-AUC y recall a precisión fija. NUNCA k-fold aleatorio.
"""
import sys
from pathlib import Path

import joblib
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

PROCESSED_DIR = Path("data/processed")
MASTER = PROCESSED_DIR / "master_ancash_ugt.parquet"
SALIDA = PROCESSED_DIR / "ancash_modelo_resultados.csv"

TARGET = "y_30"
N_SPLITS = 4
RANDOM_STATE = 42
UMBRAL_GO = 0.05  # el modelo debe superar el mejor baseline por +0.05 PR-AUC

FEATURES = [
    "inc_prot_1w", "inc_prot_4w", "inc_viol_1w", "inc_viol_4w",
    "delta_prot", "racha_prot", "dias_desde_ultima_prot",
    "n_feriados", "es_semana_electoral", "dias_hasta_eleccion", "es_fecha_critica",
    "mes", "trimestre", "semana_iso",
    "tasa_pobreza", "def_escalamiento_ancash",
]

MODELOS = {
    "logistic_regression": Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "random_forest": Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
}


def _recall_a_precision(y, s, p=0.5):
    prec, rec, _ = precision_recall_curve(y, s)
    m = prec >= p
    return float(rec[m].max()) if m.any() else 0.0


def _preparar(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = master[master[TARGET].notna()].sort_values(["semana_inicio", "ugt"]).reset_index(drop=True)
    return df[FEATURES], df[TARGET].astype(int)


def evaluar_modelo(pipeline, X, y):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    pr, rc = [], []
    for tr, te in tscv.split(X):
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        m = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        pr.append(average_precision_score(y.iloc[te], s))
        rc.append(_recall_a_precision(y.iloc[te].values, s))
    return pr, rc


def evaluar_baselines(X, y):
    """Baselines triviales bajo el mismo walk-forward."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    res = {"siempre_negativo": [], "tasa_historica": []}
    for tr, te in tscv.split(X):
        if y.iloc[te].nunique() < 2:
            continue
        yte = y.iloc[te].values
        res["siempre_negativo"].append(average_precision_score(yte, np.zeros(len(yte))))
        res["tasa_historica"].append(average_precision_score(yte, np.full(len(yte), y.iloc[tr].mean())))
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)
    X, y = _preparar(master)

    print(f"Dataset: {len(X)} filas, {y.mean()*100:.1f}% positivos ({TARGET})\n")

    base = evaluar_baselines(X, y)
    base_media = {k: float(np.mean(v)) if v else float("nan") for k, v in base.items()}
    mejor_base_nombre = max(base_media, key=base_media.get)
    mejor_base = base_media[mejor_base_nombre]

    filas = []
    print(f"{'Modelo':<22} | {'PR-AUC medio':<13} | {'±std':<8} | {'Recall@P50':<10} | {'folds':<5}")
    print("-" * 70)
    for nombre, p, r in [("siempre_negativo", base["siempre_negativo"], None),
                         ("tasa_historica", base["tasa_historica"], None)]:
        m = float(np.mean(p)) if p else float("nan")
        s = float(np.std(p)) if p else float("nan")
        print(f"{nombre:<22} | {m:<13.4f} | {s:<8.4f} | {'—':<10} | {len(p):<5}")
        filas.append({"tipo": "baseline", "nombre": nombre, "pr_auc_medio": round(m, 4), "pr_auc_std": round(s, 4)})

    mejor_modelo, mejor_modelo_pr = None, -1
    for nombre, pipe in MODELOS.items():
        pr, rc = evaluar_modelo(pipe, X, y)
        m = float(np.mean(pr)) if pr else float("nan")
        s = float(np.std(pr)) if pr else float("nan")
        rec = float(np.mean(rc)) if rc else float("nan")
        print(f"{nombre:<22} | {m:<13.4f} | {s:<8.4f} | {rec:<10.4f} | {len(pr):<5}")
        filas.append({"tipo": "modelo", "nombre": nombre, "pr_auc_medio": round(m, 4),
                      "pr_auc_std": round(s, 4), "recall_p50": round(rec, 4)})
        if not np.isnan(m) and m > mejor_modelo_pr:
            mejor_modelo, mejor_modelo_pr = nombre, m

    pd.DataFrame(filas).to_csv(SALIDA, index=False)

    dif = mejor_modelo_pr - mejor_base
    es_go = dif >= UMBRAL_GO
    veredicto = "GO ✅" if es_go else "NO-GO ❌"
    print(f"""
╔════════════════════════════════════════════════════════╗
║   PUNTO DE CONTROL — Track A regional Áncash ({TARGET})      ║
╠════════════════════════════════════════════════════════╣
  Mejor baseline:  {mejor_base_nombre} = {mejor_base:.4f}
  Mejor modelo:    {mejor_modelo} = {mejor_modelo_pr:.4f}
  Diferencia:      {dif:+.4f}   (umbral GO: +{UMBRAL_GO})
  VEREDICTO:       {veredicto}
╚════════════════════════════════════════════════════════╝
""")
    print(f"✓ Resultados guardados en {SALIDA}")

    # Si pasa el control, guardar el modelo final entrenado sobre TODOS los datos.
    # Se prefiere logistic_regression por su estabilidad (±std mucho menor que RF
    # con este dataset pequeño), aunque el PR-AUC medio sea casi idéntico.
    if es_go:
        modelo_final_nombre = "logistic_regression"
        modelo_final = clone(MODELOS[modelo_final_nombre]).fit(X, y)
        ruta = Path("models") / "modelo_v1_track_A_ancash.pkl"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": modelo_final,
            "feature_cols": FEATURES,
            "track": "A",
            "region": "Áncash",
            "unidad": "UGT × semana",
            "target": TARGET,
            "pr_auc_cv": round(mejor_modelo_pr, 4),
            "pr_auc_baseline": round(mejor_base, 4),
            "modelo_tipo": modelo_final_nombre,
            "trained_on": str(pd.Timestamp.now().date()),
        }, ruta)
        print(f"✓ Modelo final guardado en {ruta} ({modelo_final_nombre}, entrenado sobre {len(X)} filas)")


if __name__ == "__main__":
    main()
