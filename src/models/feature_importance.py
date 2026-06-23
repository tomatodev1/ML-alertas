import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from src.models.train_simple import MODELOS, TARGET, _preparar_track

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MASTER = PROCESSED_DIR / "master_zona_semana.parquet"
SALIDA = PROCESSED_DIR / "feature_importance.csv"

N_TOP = 10


# ── Cálculo ───────────────────────────────────────────────────────────────────
def calcular_importancias(master: pd.DataFrame, track: str) -> pd.Series:
    """Ajusta el random forest sobre todos los datos disponibles del track
    (no es validación, solo inspección rápida de señal por feature)."""
    X, y = _preparar_track(master, track, TARGET)
    pipeline = clone(MODELOS["random_forest"])
    pipeline.fit(X, y)
    importancias = pipeline.named_steps["clf"].feature_importances_
    return pd.Series(importancias, index=X.columns).sort_values(ascending=False)


# ── Reporte ───────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    master = pd.read_parquet(MASTER)

    imp_a = calcular_importancias(master, "A")
    imp_b = calcular_importancias(master, "B")

    todas_features = sorted(set(imp_a.index) | set(imp_b.index))
    tabla = pd.DataFrame({
        "feature": todas_features,
        "importance_track_a": [round(float(imp_a.get(f, float("nan"))), 4) for f in todas_features],
        "importance_track_b": [round(float(imp_b.get(f, float("nan"))), 4) for f in todas_features],
    })
    tabla["_orden"] = tabla[["importance_track_a", "importance_track_b"]].max(axis=1)
    tabla = tabla.sort_values("_orden", ascending=False).drop(columns="_orden").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)

    print(f"\nTop {N_TOP} features por importancia (random forest, {TARGET}):\n")
    print(f"{'Feature':<26} | {'Track A':<10} | {'Track B':<10}")
    print("-" * 52)
    for _, fila in tabla.head(N_TOP).iterrows():
        print(f"{fila['feature']:<26} | {fila['importance_track_a']:<10} | {fila['importance_track_b']:<10}")

    print(f"\n✓ Importancias guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
