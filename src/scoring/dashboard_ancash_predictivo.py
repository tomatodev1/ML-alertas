"""Dashboard predictivo de Áncash — plantilla de Claude Design + datos reales.

La plantilla visual (mapa, medidores, gráficos, tabla) viene de un proyecto de
Claude Design (`src/scoring/assets/ancash_dashboard/template.html`, runtime
`support.js`). Este script NO toca el diseño: solo calcula los datos reales
del pipeline y los inyecta vía `window.__DATA__` antes de que el componente
los renderice en el navegador.

Fuentes de cada bloque de datos:
  - Predicción por UGT (prob)      -> score_ancash.predecir() (modelo Track A)
  - Frecuencia histórica (freq)    -> data/processed/master_ancash_ugt.parquet
  - Conteo de incidentes reales    -> build_ancash._incidentes_por_ugt()
  - Tabla de detalle por distrito  -> ancash_datos.calcular_indice() (capa
    OBSERVACIONAL, no es salida del modelo — se mantiene así en el dashboard)
  - Métricas del modelo (PR-AUC)   -> metadata de models/modelo_v1_track_A_ancash.pkl
"""
import json
import shutil
import sys
import webbrowser
from pathlib import Path

import joblib
import pandas as pd

from src.dataset.build_ancash import UGTS, _incidentes_por_ugt
from src.scoring.ancash_datos import calcular_indice
from src.scoring.score_ancash import MODELO_PATH, predecir

ASSETS_DIR = Path("src/scoring/assets/ancash_dashboard")
TEMPLATE = ASSETS_DIR / "template.html"
MASTER = Path("data/processed/master_ancash_ugt.parquet")
SALIDA_DIR = Path("data/processed")
SALIDA_HTML = SALIDA_DIR / "dashboard_ancash_predictivo.html"


def _ugts_data(pred: pd.DataFrame, master_valid: pd.DataFrame, inc: pd.DataFrame) -> list[dict]:
    freq = master_valid.groupby("ugt")["y_30"].mean() * 100
    conteo = inc.groupby(["ugt", "categoria"]).size().unstack(fill_value=0)
    prob_por_ugt = pred.set_index("ugt")["probabilidad"]

    datos = []
    for ugt in UGTS:
        protesta = int(conteo.loc[ugt, "PROTESTA"]) if ugt in conteo.index and "PROTESTA" in conteo.columns else 0
        violencia = int(conteo.loc[ugt, "VIOLENCIA"]) if ugt in conteo.index and "VIOLENCIA" in conteo.columns else 0
        datos.append({
            "name": ugt,
            "prob": round(float(prob_por_ugt[ugt]) * 100, 1),
            "freq": round(float(freq.get(ugt, 0.0))),
            "protesta": protesta,
            "violencia": violencia,
        })
    return datos


def _rows_data(idx: pd.DataFrame) -> list[list]:
    """Tabla de detalle por distrito, agrupada por UGT (orden JERARQUIA) y
    ordenada por índice descendente dentro de cada UGT."""
    filas = []
    for ugt in UGTS:
        sub = idx[idx["ugt"] == ugt].sort_values("score", ascending=False)
        for _, r in sub.iterrows():
            fecha = r["ultimo_evento"].strftime("%d/%m/%Y") if pd.notna(r["ultimo_evento"]) else "—"
            filas.append([r["ugt"], r["provincia"], r["distrito"], int(r["n_total"]), fecha, int(r["score"])])
    return filas


def construir_datos() -> dict:
    master = pd.read_parquet(MASTER)
    master_valid = master[master["y_30"].notna()]
    inc = _incidentes_por_ugt()
    idx = calcular_indice()
    pred = predecir()
    meta_modelo = joblib.load(MODELO_PATH)

    pr_auc_model = float(meta_modelo["pr_auc_cv"])
    pr_auc_baseline = float(meta_modelo["pr_auc_baseline"])

    return {
        "ugts": _ugts_data(pred, master_valid, inc),
        "rows": _rows_data(idx),
        "model": {"prAucModel": pr_auc_model, "prAucBaseline": pr_auc_baseline},
        "meta": {
            "weekLabel": str(pred["semana_scoring"].iloc[0].date()),
            "nObs": int(len(master_valid)),
            "nUgts": int(master_valid["ugt"].nunique()),
            "semanasPorUgt": round(len(master_valid) / master_valid["ugt"].nunique()),
            "nPosWeeks": int(master_valid["y_30"].sum()),
            "prAucModelStr": f"{pr_auc_model:.2f}",
            "prAucBaselineStr": f"{pr_auc_baseline:.2f}",
            "prAucDiffStr": f"{pr_auc_model - pr_auc_baseline:+.2f}",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    datos = construir_datos()

    plantilla = TEMPLATE.read_text(encoding="utf-8")
    html = plantilla.replace("__DATA_JSON__", json.dumps(datos, ensure_ascii=False))

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    (SALIDA_DIR / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ASSETS_DIR / "support.js", SALIDA_DIR / "support.js")
    shutil.copyfile(ASSETS_DIR / "assets" / "logoPROTECTA.png", SALIDA_DIR / "assets" / "logoPROTECTA.png")
    SALIDA_HTML.write_text(html, encoding="utf-8")

    print(f"✓ Dashboard predictivo de Áncash guardado en {SALIDA_HTML}")
    print(f"  Semana: {datos['meta']['weekLabel']} · PR-AUC modelo {datos['meta']['prAucModelStr']} "
          f"vs baseline {datos['meta']['prAucBaselineStr']}")
    for u in datos["ugts"]:
        print(f"  {u['name']:<18} {u['prob']:5.1f}%")

    if "--no-open" not in sys.argv:
        try:
            webbrowser.open(SALIDA_HTML.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
