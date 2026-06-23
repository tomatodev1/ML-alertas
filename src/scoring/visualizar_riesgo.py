"""Genera un PNG con la evolución del riesgo predicho por zona (Track B):
backtest histórico (in-sample, el modelo final ya vio estos datos en
entrenamiento) + el/los puntos de scoring en vivo guardados en la BD."""
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # sin display disponible en este entorno
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.scoring.score_semanal import MODELO_PATH, ZONAS_TRACK_B

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

MASTER = Path("data/processed/master_zona_semana.parquet")
SALIDA_PNG = Path("data/processed/riesgo_historico_track_b.png")


# ── Datos ─────────────────────────────────────────────────────────────────────
def _cargar_historico_scoreado(paquete: dict) -> pd.DataFrame:
    """Backtest in-sample: el modelo guardado se entrenó sobre TODOS estos
    datos (Paso 7 de Fase 3), así que esta curva es optimista, no un
    walk-forward real. Útil para ver patrones, no para medir desempeño."""
    master = pd.read_parquet(MASTER)
    df = master[master["track"] == "B"].copy()
    df = df.sort_values(["zona", "semana_inicio"]).reset_index(drop=True)

    X = df[paquete["feature_cols"]].fillna(0)
    df["probabilidad"] = paquete["model"].predict_proba(X)[:, 1]
    return df[["zona", "semana_inicio", "probabilidad", "y_30"]]


def _cargar_scoring_en_vivo() -> pd.DataFrame:
    columnas = ["zona", "semana_inicio", "probabilidad"]
    if not DATABASE_URL:
        return pd.DataFrame(columns=columnas)
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT zona, semana_scoring AS semana_inicio, probabilidad FROM riesgo_zona_semana"),
            conn,
        )
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"])
    return df


# ── Gráfico ───────────────────────────────────────────────────────────────────
def graficar(df_hist: pd.DataFrame, df_vivo: pd.DataFrame, umbral: float) -> None:
    fig, axes = plt.subplots(len(ZONAS_TRACK_B), 1, figsize=(11, 14), sharex=True)

    for ax, zona in zip(axes, ZONAS_TRACK_B):
        sub_h = df_hist[df_hist["zona"] == zona]
        sub_v = df_vivo[df_vivo["zona"] == zona]

        ax.plot(sub_h["semana_inicio"], sub_h["probabilidad"], color="steelblue", lw=1.2, label="histórico (backtest in-sample)")
        positivos = sub_h[sub_h["y_30"] == 1]
        ax.scatter(positivos["semana_inicio"], positivos["probabilidad"], color="crimson", zorder=5, s=20, label="conflicto real (y_30=1)")

        if not sub_v.empty:
            ax.plot(sub_v["semana_inicio"], sub_v["probabilidad"], color="darkorange", marker="o", lw=0, markersize=8, label="scoring en vivo")

        ax.axhline(umbral, color="gray", linestyle="--", lw=1, label=f"umbral={umbral}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(zona, fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Riesgo de conflicto predicho (P(y_30)) — Track B, por zona", fontsize=13)
    fig.tight_layout()

    SALIDA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA_PNG, dpi=130)
    plt.close(fig)


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    paquete = joblib.load(MODELO_PATH)

    df_hist = _cargar_historico_scoreado(paquete)
    df_vivo = _cargar_scoring_en_vivo()
    graficar(df_hist, df_vivo, paquete["threshold"])

    print(f"✓ Gráfico guardado en {SALIDA_PNG}")
    print("  Nota: la curva histórica es backtest in-sample (el modelo final ya vio")
    print("  estos datos en entrenamiento) — sirve para ver patrones, no como medida")
    print(f"  de desempeño real (esa es PR-AUC={paquete['pr_auc_cv']:.3f} en CV walk-forward).")


if __name__ == "__main__":
    main()
