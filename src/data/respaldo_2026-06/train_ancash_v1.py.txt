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
    "tasa_pobreza", "def_escalamiento_ancash", "oefa_denuncias_mineria_12m",
    "rep_antamina_neg_4w", "rep_compromiso_4w",
]
# "cobre_precio_usd", "cobre_ret_4w" (precio del cobre HG=F COMEX, loader_cobre.py)
# se probaron y descartaron: PR-AUC 0.8004 → 0.7989 (−0.0015). En la ventana
# 2024–2026 el precio del cobre tiene tendencia monotónica al alza (~4→~6 USD/lb)
# que se solapa con las features de calendario (mes, trimestre, semana_iso) — el
# modelo no distingue señal causal de correlación temporal espuria. Con una
# ventana de entrenamiento más larga (ciclos completos de commodity) el resultado
# podría ser distinto. Código disponible en _join_cobre() de build_ancash.
# "prot_impacto_ultimo" (severidad de la última protesta, dataProtestas.xlsx)
# se probó y se descartó: bajó el PR-AUC de 0.780 a 0.761 (logistic_regression)
# "hist_prot_antamina_acum", "hist_prot_antamina_5y" (tensión histórica 2001–,
# dataHistoricaProtestas.xlsx) se probaron y se descartaron: bajaron PR-AUC de
# 0.8004 a 0.7962. Son casi constantes por UGT en la ventana 2024-2026 (todos
# los eventos históricos son pre-2024), por lo que actúan como ruido. Cuando
# dataIncidentes cubra períodos anteriores a 2023, podrían activarse junto con
# una ventana de entrenamiento extendida. Ver _join_historica() en build_ancash.
# en vez de mejorarlo. La función _join_impacto() en build_ancash.py se
# mantiene (la columna queda en el dataset) por si sirve en una iteración
# futura con más datos, pero no se usa en el modelo. Ver memoria del proyecto.
# "def_conf_activos_ugt", "def_conf_antamina_ugt" (cronicidad de conflictos por
# UGT desde los Reportes Mensuales de la Defensoría, 2016-2026;
# _join_defensoria_conflictos en build_ancash) se probaron y se descartaron:
# delta PR-AUC +0.0000 (LR) / +0.0018 (RF) sobre la ventana 2024-2026. Como
# hist_prot y OEFA, la cronicidad mensual es redundante con las features
# autoregresivas de corto plazo. El VALOR de esa fuente Defensoría no es como
# feature de contexto sino como CIMIENTO para extender la ventana de
# entrenamiento hacia atrás (2016-2023 como filas reales de label). Ver memoria
# del proyecto y src.data.loader_defensoria_historico.

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
    pr_auc_por_modelo: dict[str, float] = {}
    for nombre, pipe in MODELOS.items():
        pr, rc = evaluar_modelo(pipe, X, y)
        m = float(np.mean(pr)) if pr else float("nan")
        s = float(np.std(pr)) if pr else float("nan")
        rec = float(np.mean(rc)) if rc else float("nan")
        print(f"{nombre:<22} | {m:<13.4f} | {s:<8.4f} | {rec:<10.4f} | {len(pr):<5}")
        filas.append({"tipo": "modelo", "nombre": nombre, "pr_auc_medio": round(m, 4),
                      "pr_auc_std": round(s, 4), "recall_p50": round(rec, 4)})
        pr_auc_por_modelo[nombre] = m
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
    # Se prefiere logistic_regression: con FECHA_INICIO=2024 (dataset actual) da
    # PR-AUC 0.8004 ±std bajo y probabilidades bien dispersas (Huarmey 90%,
    # Huallanca 25%). RF mejora en ranking pero sus probabilidades quedan
    # comprimidas sin calibración efectiva — ver comentario en build_ancash.py.
    if es_go:
        modelo_final_nombre = "logistic_regression"
        modelo_final = clone(MODELOS[modelo_final_nombre]).fit(X, y)
        # pr_auc_cv: del modelo que efectivamente se guarda (LR), no del mejor
        # candidato del go/no-go (que puede ser RF con PR-AUC ligeramente mayor).
        pr_auc_modelo_final = pr_auc_por_modelo[modelo_final_nombre]
        ruta = Path("models") / "modelo_v1_track_A_ancash.pkl"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": modelo_final,
            "feature_cols": FEATURES,
            "track": "A",
            "region": "Áncash",
            "unidad": "UGT × semana",
            "target": TARGET,
            "pr_auc_cv": round(pr_auc_modelo_final, 4),
            "pr_auc_mejor_candidato": round(mejor_modelo_pr, 4),
            "pr_auc_baseline": round(mejor_base, 4),
            "modelo_tipo": modelo_final_nombre,
            "trained_on": str(pd.Timestamp.now().date()),
        }, ruta)
        print(f"✓ Modelo final guardado en {ruta} ({modelo_final_nombre}, PR-AUC={pr_auc_modelo_final:.4f}, "
              f"entrenado sobre {len(X)} filas)")


if __name__ == "__main__":
    main()
