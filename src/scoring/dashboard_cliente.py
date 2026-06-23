"""Dashboard ejecutivo orientado al cliente (no técnico).
Muestra el riesgo de conflicto social por zona con medidores tipo velocímetro,
tarjetas resumen y una tendencia clara. Salida: un HTML standalone con fondo
azul marino degradado, listo para presentar.
"""
import sys
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit

from src.models.train_xgboost import _modelo_xgb, _scale_pos_weight
from src.scoring.score_semanal import MODELO_PATH, ZONA_CLIENTE, ZONAS_TRACK_B
from src.scoring.visualizar_riesgo import _cargar_historico_scoreado, _cargar_scoring_en_vivo

MASTER = Path("data/processed/master_zona_semana.parquet")

SALIDA_HTML = Path("data/processed/dashboard_cliente.html")
PREVIEW_PNG = Path("data/processed/_preview_dashboard.png")

# ── Paleta ────────────────────────────────────────────────────────────────────
VERDE = "#27ae60"
AMBAR = "#f4c430"
ROJO = "#e74c3c"
TEXTO = "#eaf1fb"
TEXTO_TENUE = "#9fb3d4"

# Umbrales de interpretación en porcentaje (el umbral de alerta del modelo es 0.55)
UMBRAL_BAJO = 35
UMBRAL_ALERTA = 55


def _estado(pct: float) -> tuple[str, str, str]:
    """Devuelve (etiqueta, color, emoji) según el nivel de riesgo."""
    if pct >= UMBRAL_ALERTA:
        return "ALERTA", ROJO, "🔴"
    if pct >= UMBRAL_BAJO:
        return "Moderado", AMBAR, "🟡"
    return "Bajo", VERDE, "🟢"


# ── Datos ─────────────────────────────────────────────────────────────────────
def _historico_oos(paquete: dict, n_splits: int = 5) -> pd.DataFrame:
    """Predicciones out-of-sample (walk-forward) sobre la historia de Track B.
    Cada semana se predice con un modelo que NO la vio en entrenamiento — curva
    honesta (a diferencia del backtest in-sample, que se ve sobreseguro)."""
    feature_cols = paquete["feature_cols"]
    master = pd.read_parquet(MASTER)
    df = master[master["track"] == "B"].sort_values(["semana_inicio", "zona"]).reset_index(drop=True)
    X = df[feature_cols].fillna(0)
    y = df["y_30"].astype(int)

    oos = pd.Series(np.nan, index=df.index)
    for idx_tr, idx_te in TimeSeriesSplit(n_splits=n_splits).split(X):
        if y.iloc[idx_tr].nunique() < 2:
            continue
        modelo = _modelo_xgb(_scale_pos_weight(y.iloc[idx_tr]))
        modelo.fit(X.iloc[idx_tr], y.iloc[idx_tr], eval_set=[(X.iloc[idx_te], y.iloc[idx_te])], verbose=False)
        oos.iloc[idx_te] = modelo.predict_proba(X.iloc[idx_te])[:, 1]

    df["probabilidad"] = oos
    return df[["zona", "semana_inicio", "probabilidad", "y_30"]].dropna(subset=["probabilidad"])


def _riesgo_actual(df_vivo: pd.DataFrame, df_hist: pd.DataFrame) -> pd.DataFrame:
    """Una fila por zona con la probabilidad más reciente disponible.
    Prioriza el scoring en vivo (BD); si una zona no tiene, usa el último
    histórico como respaldo."""
    filas = []
    for zona in ZONAS_TRACK_B:
        sv = df_vivo[df_vivo["zona"] == zona].sort_values("semana_inicio")
        if not sv.empty:
            prob = float(sv.iloc[-1]["probabilidad"])
            fecha = sv.iloc[-1]["semana_inicio"]
        else:
            sh = df_hist[df_hist["zona"] == zona].sort_values("semana_inicio")
            prob = float(sh.iloc[-1]["probabilidad"]) if not sh.empty else 0.0
            fecha = sh.iloc[-1]["semana_inicio"] if not sh.empty else pd.NaT
        filas.append({"zona": zona, "probabilidad": prob, "fecha": fecha})
    return pd.DataFrame(filas)


# ── Figura 1: medidores (velocímetros) ────────────────────────────────────────
def figura_medidores(df_actual: pd.DataFrame) -> go.Figure:
    n = len(ZONAS_TRACK_B)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "indicator"}] * n],
        horizontal_spacing=0.04,
    )

    for i, (_, fila) in enumerate(df_actual.iterrows()):
        pct = fila["probabilidad"] * 100
        etiqueta, color, emoji = _estado(pct)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 30, "color": TEXTO}},
                title={
                    "text": f"<b>{fila['zona']}</b><br>"
                            f"<span style='font-size:0.75em;color:{color}'>{emoji} {etiqueta}</span>",
                    "font": {"size": 15, "color": TEXTO},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": TEXTO_TENUE,
                        "tickfont": {"color": TEXTO_TENUE, "size": 9},
                        "ticksuffix": "%",
                    },
                    "bar": {"color": "rgba(255,255,255,0.92)", "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,0.04)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, UMBRAL_BAJO], "color": "rgba(39,174,96,0.55)"},
                        {"range": [UMBRAL_BAJO, UMBRAL_ALERTA], "color": "rgba(244,196,48,0.55)"},
                        {"range": [UMBRAL_ALERTA, 100], "color": "rgba(231,76,60,0.6)"},
                    ],
                    "threshold": {
                        "line": {"color": TEXTO, "width": 3},
                        "thickness": 0.85,
                        "value": UMBRAL_ALERTA,
                    },
                },
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXTO, "family": "Segoe UI, Arial, sans-serif"},
        height=300,
        margin=dict(t=70, b=10, l=20, r=20),
    )
    return fig


# ── Figura 2: ranking de barras (comparación clara entre zonas) ──────────────
def figura_ranking(df_actual: pd.DataFrame) -> go.Figure:
    df = df_actual.sort_values("probabilidad").copy()  # menor abajo, mayor arriba
    pcts = df["probabilidad"] * 100
    colores = [_estado(p)[1] for p in pcts]
    etiquetas = [f"{_estado(p)[2]} {p:.0f}%" for p in pcts]

    fig = go.Figure(go.Bar(
        x=pcts, y=df["zona"], orientation="h",
        marker=dict(color=colores, line=dict(color="rgba(255,255,255,0.25)", width=1)),
        text=etiquetas, textposition="outside",
        textfont=dict(color=TEXTO, size=14),
        hovertemplate="<b>%{y}</b><br>Riesgo: %{x:.0f}%<extra></extra>",
        cliponaxis=False,
    ))
    fig.add_vline(
        x=UMBRAL_ALERTA, line=dict(color=TEXTO, dash="dash", width=2),
        annotation_text="Umbral de alerta", annotation_position="top",
        annotation_font_color=TEXTO_TENUE,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font={"color": TEXTO, "family": "Segoe UI, Arial, sans-serif"},
        height=300, margin=dict(t=30, b=30, l=20, r=40),
        bargap=0.45,
    )
    fig.update_xaxes(range=[0, 112], ticksuffix="%", gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(tickfont=dict(size=14))
    return fig


# ── Figura 3: tendencia ───────────────────────────────────────────────────────
def figura_tendencia(df_hist: pd.DataFrame, df_vivo: pd.DataFrame) -> go.Figure:
    colores = {
        "Ica": "#4da3ff", "Pisco": "#ffb14d", "Huarmey": "#5ad19a",
        "Barranca": "#c79bff", "Lima Provincias": "#ff9aa2",
    }
    fig = go.Figure()

    # Banda de alerta (sombra roja por encima del umbral)
    fig.add_hrect(
        y0=UMBRAL_ALERTA / 100, y1=1.0,
        fillcolor="rgba(231,76,60,0.12)", line_width=0, layer="below",
    )

    for zona in ZONAS_TRACK_B:
        sub_h = df_hist[df_hist["zona"] == zona].sort_values("semana_inicio")
        sub_v = df_vivo[df_vivo["zona"] == zona]
        color = colores.get(zona, "#4da3ff")

        fig.add_trace(go.Scatter(
            x=sub_h["semana_inicio"], y=sub_h["probabilidad"],
            mode="lines", name=zona, legendgroup=zona,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{zona}</b><br>%{{x|%d %b %Y}}<br>Riesgo: %{{y:.0%}}<extra></extra>",
        ))
        if not sub_v.empty:
            fig.add_trace(go.Scatter(
                x=sub_v["semana_inicio"], y=sub_v["probabilidad"],
                mode="markers", name=f"{zona} (hoy)", legendgroup=zona, showlegend=False,
                marker=dict(color=color, size=13, symbol="star", line=dict(color="white", width=1)),
                hovertemplate=f"<b>{zona} — scoring actual</b><br>%{{x|%d %b %Y}}<br>Riesgo: %{{y:.0%}}<extra></extra>",
            ))

    fig.add_hline(
        y=UMBRAL_ALERTA / 100, line=dict(color=TEXTO_TENUE, dash="dash", width=1.5),
        annotation_text="Umbral de alerta (55%)", annotation_position="top left",
        annotation_font_color=TEXTO_TENUE,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font={"color": TEXTO, "family": "Segoe UI, Arial, sans-serif"},
        height=460,
        margin=dict(t=20, b=20, l=55, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                    font=dict(color=TEXTO, size=12)),
    )
    fig.update_yaxes(
        range=[-0.03, 1.03], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)",
        zeroline=False, title_text="Probabilidad de conflicto",
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(255,255,255,0.04)"),
    )
    return fig


# ── Tarjetas KPI (HTML) ───────────────────────────────────────────────────────
def _tarjetas_kpi(df_actual: pd.DataFrame) -> str:
    n_total = len(df_actual)
    n_alerta = int((df_actual["probabilidad"] * 100 >= UMBRAL_ALERTA).sum())
    prom = df_actual["probabilidad"].mean() * 100
    top = df_actual.loc[df_actual["probabilidad"].idxmax()]
    top_pct = top["probabilidad"] * 100
    _, top_color, _ = _estado(top_pct)
    color_alerta = ROJO if n_alerta > 0 else VERDE

    def card(valor, etiqueta, color=TEXTO):
        return f"""
        <div class="kpi-card">
          <div class="kpi-valor" style="color:{color}">{valor}</div>
          <div class="kpi-etiqueta">{etiqueta}</div>
        </div>"""

    return (
        card(n_total, "Zonas monitoreadas")
        + card(n_alerta, "Zonas en alerta", color_alerta)
        + card(f"{prom:.0f}%", "Riesgo promedio")
        + card(f"{top['zona']}", f"Mayor riesgo · {top_pct:.0f}%", top_color)
    )


def _filas_clientes(df_actual: pd.DataFrame) -> str:
    filas = ""
    for _, f in df_actual.sort_values("probabilidad", ascending=False).iterrows():
        pct = f["probabilidad"] * 100
        etiqueta, color, emoji = _estado(pct)
        clientes = ", ".join(ZONA_CLIENTE.get(f["zona"], [])) or "—"
        filas += f"""
        <tr>
          <td><b>{f['zona']}</b></td>
          <td>{clientes}</td>
          <td style="text-align:right">{pct:.0f}%</td>
          <td style="color:{color};text-align:center">{emoji} {etiqueta}</td>
        </tr>"""
    return filas


# ── Ensamblado HTML ───────────────────────────────────────────────────────────
def construir_html(df_actual, fig_medidores, fig_ranking, fig_tendencia) -> str:
    fecha_ref = df_actual["fecha"].max()
    fecha_txt = pd.Timestamp(fecha_ref).strftime("%d de %B de %Y") if pd.notna(fecha_ref) else "—"

    div_medidores = fig_medidores.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    div_ranking = fig_ranking.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    div_tendencia = fig_tendencia.to_html(
        full_html=False, include_plotlyjs=False,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sistema de Alerta Temprana — Riesgo de Conflictos</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 0 50px 0;
    font-family: "Segoe UI", Arial, sans-serif;
    color: {TEXTO};
    background: linear-gradient(135deg, #06122e 0%, #0d2350 45%, #0a1c40 75%, #061130 100%);
    background-attachment: fixed;
    min-height: 100vh;
  }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 0 24px; }}
  header {{ padding: 34px 0 10px 0; }}
  h1 {{ margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.3px; }}
  .sub {{ color: {TEXTO_TENUE}; font-size: 15px; margin-top: 6px; }}
  .badge {{
    display:inline-block; margin-top:14px; padding:6px 14px; border-radius:20px;
    background: rgba(77,163,255,0.15); border:1px solid rgba(77,163,255,0.35);
    color:#bcd6ff; font-size:13px;
  }}
  .kpi-row {{ display:grid; grid-template-columns: repeat(4,1fr); gap:16px; margin:24px 0; }}
  .kpi-card {{
    background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
    border-radius:16px; padding:20px; text-align:center;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
  }}
  .kpi-valor {{ font-size:34px; font-weight:700; line-height:1; }}
  .kpi-etiqueta {{ color:{TEXTO_TENUE}; font-size:13px; margin-top:8px; }}
  .panel {{
    background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10);
    border-radius:18px; padding:18px 18px 8px 18px; margin:20px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.22);
  }}
  .panel h2 {{ margin:4px 4px 6px 4px; font-size:18px; font-weight:600; }}
  .panel .hint {{ color:{TEXTO_TENUE}; font-size:13px; margin:0 4px 10px 4px; }}
  .leyenda {{ display:flex; gap:22px; flex-wrap:wrap; margin:6px 4px 14px 4px; font-size:13px; color:{TEXTO_TENUE}; }}
  table {{ width:100%; border-collapse:collapse; margin-top:6px; font-size:14px; }}
  th {{ text-align:left; color:{TEXTO_TENUE}; font-weight:600; padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.12); }}
  td {{ padding:11px 12px; border-bottom:1px solid rgba(255,255,255,0.06); }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  footer {{ color:{TEXTO_TENUE}; font-size:12px; text-align:center; margin-top:30px; line-height:1.6; }}
  @media (max-width: 820px) {{ .kpi-row {{ grid-template-columns: repeat(2,1fr); }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Sistema de Alerta Temprana de Conflictos Sociales</h1>
    <div class="sub">Riesgo estimado de protesta o conflicto por zona · Próximos 30 días</div>
    <div class="badge">📅 Semana evaluada: {fecha_txt}</div>
  </header>

  <div class="kpi-row">{_tarjetas_kpi(df_actual)}</div>

  <div class="panel">
    <h2>Nivel de riesgo actual por zona</h2>
    <p class="hint">Cada medidor indica la probabilidad de que ocurra un conflicto en los próximos 30 días.
       La línea blanca marca el umbral de alerta.</p>
    <div class="leyenda">
      <span>🟢 <b>Bajo</b> (0–35%)</span>
      <span>🟡 <b>Moderado</b> (35–55%)</span>
      <span>🔴 <b>Alto / Alerta</b> (55%+)</span>
    </div>
    {div_medidores}
  </div>

  <div class="panel">
    <h2>Comparación de zonas — ¿cuáles requieren atención?</h2>
    <p class="hint">Ranking de mayor a menor riesgo esta semana. La línea punteada es el umbral de alerta:
       todo lo que la supera (barras rojas) requiere atención prioritaria.</p>
    {div_ranking}
  </div>

  <div class="panel">
    <h2>¿Cómo ha evolucionado el riesgo?</h2>
    <p class="hint">Estimación semana a semana (validación honesta: cada punto se predijo sin haber visto esa semana).
       Pasa el cursor para ver valores, arrastra la barra inferior para enfocar un periodo. La franja roja
       superior es la zona de alerta · ⭐ marca el dato de esta semana.</p>
    {div_tendencia}
  </div>

  <div class="panel">
    <h2>Detalle por zona y cliente</h2>
    <table>
      <thead><tr><th>Zona</th><th>Cliente(s)</th><th style="text-align:right">Riesgo</th><th style="text-align:center">Estado</th></tr></thead>
      <tbody>{_filas_clientes(df_actual)}</tbody>
    </table>
  </div>

  <footer>
    Modelo predictivo de conflictos sociales · PROTECTA PERÚ &nbsp;·&nbsp; Horizonte: 30 días &nbsp;·&nbsp; Generado automáticamente<br>
    Estimación estadística de apoyo a la decisión; complementa —no reemplaza— el sistema de alertas y el criterio del analista.
  </footer>
</div>
</body>
</html>"""


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    paquete = joblib.load(MODELO_PATH)

    # Riesgo actual (gauges/ranking/tabla): scoring en vivo de la BD.
    df_vivo = _cargar_scoring_en_vivo()
    df_hist_insample = _cargar_historico_scoreado(paquete)  # solo respaldo si faltara una zona
    df_actual = _riesgo_actual(df_vivo, df_hist_insample)

    # Tendencia histórica: predicciones out-of-sample (honestas), no in-sample.
    df_hist_oos = _historico_oos(paquete)

    fig_medidores = figura_medidores(df_actual)
    fig_ranking = figura_ranking(df_actual)
    fig_tendencia = figura_tendencia(df_hist_oos, df_vivo)

    html = construir_html(df_actual, fig_medidores, fig_ranking, fig_tendencia)
    SALIDA_HTML.parent.mkdir(parents=True, exist_ok=True)
    SALIDA_HTML.write_text(html, encoding="utf-8")

    print(f"✓ Dashboard de cliente guardado en {SALIDA_HTML}")
    print(f"  Zonas en alerta: {int((df_actual['probabilidad']*100 >= UMBRAL_ALERTA).sum())}/{len(df_actual)}")

    # Vista previa PNG (solo para revisión interna)
    if "--preview" in sys.argv:
        try:
            fig_ranking.write_image(str(PREVIEW_PNG.with_name('_preview_ranking.png')), width=1300, height=300, scale=2)
            fig_tendencia.update_layout(paper_bgcolor="#0d2350").write_image(
                str(PREVIEW_PNG.with_name('_preview_tendencia2.png')), width=1300, height=460, scale=2)
            print("  Vistas previas PNG generadas (ranking + tendencia).")
        except Exception as e:
            print(f"  (No se pudo generar PNG de preview: {e})")

    if "--no-open" not in sys.argv:
        try:
            webbrowser.open(SALIDA_HTML.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
