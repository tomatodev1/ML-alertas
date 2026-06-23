"""Dashboard interactivo (Plotly, HTML standalone) con la evolución del riesgo
predicho por zona (Track B): backtest histórico (in-sample) + scoring en vivo.
Zoom, pan, hover con valores exactos y selector de rango de fechas."""
import sys
import webbrowser
from pathlib import Path

import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.scoring.score_semanal import MODELO_PATH, ZONAS_TRACK_B
from src.scoring.visualizar_riesgo import _cargar_historico_scoreado, _cargar_scoring_en_vivo

SALIDA_HTML = Path("data/processed/riesgo_historico_track_b.html")

COLORES = {
    "Ica": "#1f77b4",
    "Pisco": "#ff7f0e",
    "Huarmey": "#2ca02c",
    "Barranca": "#9467bd",
    "Lima Provincias": "#8c564b",
}


# ── Figura ────────────────────────────────────────────────────────────────────
def construir_figura(df_hist, df_vivo, umbral: float) -> go.Figure:
    n = len(ZONAS_TRACK_B)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=ZONAS_TRACK_B,
        vertical_spacing=0.04,
    )

    for i, zona in enumerate(ZONAS_TRACK_B, start=1):
        color = COLORES.get(zona, "#1f77b4")
        sub_h = df_hist[df_hist["zona"] == zona]
        sub_v = df_vivo[df_vivo["zona"] == zona]
        positivos = sub_h[sub_h["y_30"] == 1]

        fig.add_trace(
            go.Scatter(
                x=sub_h["semana_inicio"], y=sub_h["probabilidad"],
                mode="lines", line=dict(color=color, width=1.6),
                name=zona, legendgroup=zona, showlegend=False,
                hovertemplate=f"<b>{zona}</b><br>%{{x|%Y-%m-%d}}<br>P(y_30)=%{{y:.3f}}<extra></extra>",
            ),
            row=i, col=1,
        )

        if not positivos.empty:
            fig.add_trace(
                go.Scatter(
                    x=positivos["semana_inicio"], y=positivos["probabilidad"],
                    mode="markers", marker=dict(color="crimson", size=7),
                    name="conflicto real (y_30=1)", legendgroup="real", showlegend=(i == 1),
                    hovertemplate=f"<b>{zona}</b><br>%{{x|%Y-%m-%d}}<br>Conflicto real confirmado<extra></extra>",
                ),
                row=i, col=1,
            )

        if not sub_v.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub_v["semana_inicio"], y=sub_v["probabilidad"],
                    mode="markers", marker=dict(color="darkorange", size=12, symbol="star"),
                    name="scoring en vivo", legendgroup="vivo", showlegend=(i == 1),
                    hovertemplate=f"<b>{zona}</b><br>%{{x|%Y-%m-%d}}<br>Scoring en vivo: P(y_30)=%{{y:.3f}}<extra></extra>",
                ),
                row=i, col=1,
            )

        fig.add_hline(
            y=umbral, line=dict(color="gray", dash="dash", width=1),
            row=i, col=1,
            annotation_text=f"umbral={umbral}" if i == 1 else None,
            annotation_position="top left",
        )
        fig.update_yaxes(range=[-0.05, 1.05], tickformat=".0%", row=i, col=1)

    fig.update_layout(
        title="Riesgo de conflicto predicho (P(y_30)) — Track B, por zona (interactivo)",
        height=235 * n,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(t=110, b=40),
        template="plotly_white",
    )

    # Selector de rango rápido (arriba) y range slider (abajo) — el zoom se
    # propaga a todos los paneles porque comparten eje x (shared_xaxes=True).
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1a", step="year", stepmode="backward"),
            dict(step="all", label="Todo"),
        ]),
        row=1, col=1,
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=n, col=1)

    return fig


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    paquete = joblib.load(MODELO_PATH)

    df_hist = _cargar_historico_scoreado(paquete)
    df_vivo = _cargar_scoring_en_vivo()

    fig = construir_figura(df_hist, df_vivo, paquete["threshold"])

    SALIDA_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(SALIDA_HTML, include_plotlyjs="cdn")

    print(f"✓ Dashboard interactivo guardado en {SALIDA_HTML}")
    print("  Nota: la curva histórica es backtest in-sample (igual que en visualizar_riesgo.py);")
    print(f"  el desempeño real de validación es PR-AUC={paquete['pr_auc_cv']:.3f} (walk-forward).")

    try:
        webbrowser.open(SALIDA_HTML.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()
