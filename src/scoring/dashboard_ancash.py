"""Dashboard de Conflictos Sociales — Región Áncash (ANTAMINA).

Primer entregable enfocado en Áncash, con la estética del prototipo guía.
Enfoque HÍBRIDO:
  · Datos REALES donde existen: índice de actividad observada por distrito
    (eventos geolocalizados de la BD interna), serie temporal y composición.
  · Secciones marcadas explícitamente como "pendiente de fuente" para lo que
    el proyecto aún no tiene (drivers socioeconómicos, compromiso de partes,
    simulador causal).

IMPORTANTE: el índice por distrito es ACTIVIDAD OBSERVADA (lo que ya ocurrió),
no una predicción del modelo ML. Áncash minero es Track A y no cuenta con
modelo predictivo desplegado.
"""
import json
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.scoring.ancash_datos import (
    calcular_indice,
    composicion_categorias,
    serie_mensual,
)
from src.scoring.score_ancash import MODELO_PATH, predecir

SALIDA_HTML = Path("data/processed/dashboard_ancash.html")

# ── Paleta (navy + teal, según el prototipo guía) ─────────────────────────────
TEAL = "#2bd4c4"
TEAL_TENUE = "#7fe3da"
TEXTO = "#e8f1f5"
TEXTO_TENUE = "#8fa8b8"
VERDE, AMBAR, NARANJA, ROJO = "#37c98a", "#f4c430", "#f0863e", "#e74d6b"


# ── Figura: actividad en el tiempo ────────────────────────────────────────────
def figura_serie(serie: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie["mes"], y=serie["eventos"],
        mode="lines", line=dict(color=TEAL, width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(43,212,196,0.12)",
        hovertemplate="%{x|%b %Y}<br>%{y} eventos<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXTO, "family": "Segoe UI, Arial, sans-serif"},
        height=300, margin=dict(t=10, b=30, l=40, r=15),
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, title_text="Eventos / mes")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
    return fig


# ── Figura: composición de conflictividad (donut real) ───────────────────────
def figura_composicion(comp: pd.DataFrame) -> go.Figure:
    comp = comp[comp["eventos"] > 0].sort_values("eventos", ascending=False)
    colores = [TEAL, NARANJA, AMBAR, ROJO][: len(comp)]

    fig = go.Figure(go.Pie(
        labels=comp["categoria"], values=comp["eventos"],
        hole=0.58, sort=False,
        marker=dict(colors=colores, line=dict(color="#0a1c2b", width=2)),
        textinfo="percent", textfont=dict(color="#06141c", size=13, family="Segoe UI"),
        hovertemplate="%{label}<br>%{value} eventos (%{percent})<extra></extra>",
    ))
    total = int(comp["eventos"].sum())
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXTO, "family": "Segoe UI, Arial, sans-serif", "size": 12},
        height=300, margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=11)),
        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:11px;color:{TEXTO_TENUE}'>eventos</span>",
                          x=0.5, y=0.5, showarrow=False, font=dict(color=TEXTO, size=22))],
    )
    return fig


# ── Capa predictiva (modelo Track A) ─────────────────────────────────────────
def _nivel_prob(pct: float) -> tuple[str, str]:
    if pct >= 55:
        return "Alto", ROJO
    if pct >= 35:
        return "Medio", AMBAR
    return "Bajo", VERDE


def _obtener_prediccion():
    """Devuelve (df_pred, metadata) o (None, None) si el modelo no está disponible."""
    if not MODELO_PATH.exists():
        return None, None
    try:
        import joblib
        pred = predecir()
        meta = joblib.load(MODELO_PATH)
        return pred, meta
    except Exception as e:
        print(f"  ⚠ No se pudo generar la predicción por UGT: {e}")
        return None, None


def _tarjetas_prediccion(pred) -> str:
    tarjetas = ""
    for _, r in pred.iterrows():
        pct = r["probabilidad"] * 100
        nivel, color = _nivel_prob(pct)
        tarjetas += f"""
        <div class="pcard">
          <div class="pc-ugt">{r['ugt']}</div>
          <div class="pc-val" style="color:{color}">{pct:.0f}<span class="pc-pct">%</span></div>
          <div class="pc-barra"><i style="width:{pct:.0f}%;background:{color}"></i></div>
          <div class="pc-nivel" style="color:{color}">{nivel}</div>
        </div>"""
    return tarjetas


# ── KPIs y chips ──────────────────────────────────────────────────────────────
def _estado_regional(score: float) -> tuple[str, str]:
    if score >= 55:
        return "Alto", ROJO
    if score >= 35:
        return "Medio", AMBAR
    return "Bajo", VERDE


def construir_html(idx: pd.DataFrame, fig_serie: go.Figure, fig_radar: go.Figure, comp: pd.DataFrame,
                   pred=None, meta=None) -> str:
    indice_reg = round(float(idx["score"].mean()))
    estado_reg, color_reg = _estado_regional(indice_reg)
    n_alerta = int((idx["score"] >= 55).sum())
    fecha_ref = pd.Timestamp.today().strftime("%d de %B de %Y")

    # Sección de predicción (capa del modelo Track A). Si no hay modelo, se omite.
    if pred is not None and not pred.empty:
        sem_pred = pd.Timestamp(pred["semana_scoring"].iloc[0]).strftime("%d/%m/%Y")
        pr_auc = meta.get("pr_auc_cv", "—") if meta else "—"
        seccion_prediccion = f"""
  <div class="panel pred-panel">
    <h2>🔮 Predicción de conflicto por UGT <span class="badge-modelo">MODELO PREDICTIVO · PR-AUC {pr_auc}</span></h2>
    <p class="hint">Probabilidad estimada de una <b>protesta nueva en los próximos 30 días</b> por unidad de gestión territorial.
       A diferencia del resto del tablero (actividad observada), esto es una <b>predicción</b> del modelo validado para Áncash ·
       semana base {sem_pred}.</p>
    <div class="pred-grid">{_tarjetas_prediccion(pred)}</div>
  </div>"""
    else:
        seccion_prediccion = """
  <div class="panel pred-panel">
    <h2>🔮 Predicción de conflicto por UGT <span class="badge-pend">MODELO NO DISPONIBLE</span></h2>
    <p class="hint">Entrena el modelo con <code>python -m src.models.train_ancash</code> para activar esta capa.</p>
  </div>"""

    # Chips de tipos de conflictividad reales dominantes
    comp_ord = comp.sort_values("eventos", ascending=False)
    chips = "".join(
        f'<span class="chip">{r["categoria"]} · {r["eventos"]}</span>'
        for _, r in comp_ord.iterrows() if r["eventos"] > 0
    )

    # Datos de distritos embebidos para el JS (filtros + render dinámico)
    distritos_js = json.dumps([
        {
            "ugt": r["ugt"], "provincia": r["provincia"], "distrito": r["distrito"],
            "score": int(r["score"]), "estado": r["estado"], "color": r["color"],
            "accion": r["accion"], "n_total": int(r["n_total"]),
            "n_protesta": int(r["n_protesta"]), "n_violencia": int(r["n_violencia"]),
            "ultimo": (r["ultimo_evento"].strftime("%d/%m/%Y") if pd.notna(r["ultimo_evento"]) else "—"),
        }
        for _, r in idx.iterrows()
    ], ensure_ascii=False)

    div_serie = fig_serie.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    div_radar = fig_radar.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conflictos Sociales — Áncash · PROTECTA PERÚ</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin:0; padding:0 0 40px 0; color:{TEXTO};
    font-family:"Segoe UI", Arial, sans-serif;
    background: radial-gradient(1200px 600px at 70% -10%, #0e2b3a 0%, transparent 60%),
                linear-gradient(160deg, #07131d 0%, #0a1c2b 55%, #081521 100%);
    background-attachment: fixed; min-height:100vh;
  }}
  .wrap {{ max-width:1300px; margin:0 auto; padding:0 22px; }}
  .topbar {{ display:flex; align-items:center; justify-content:space-between; padding:20px 0 6px; }}
  .brand {{ display:flex; align-items:center; gap:14px; }}
  .logo {{ background:linear-gradient(135deg,{TEAL},#1b8f9c); color:#04121a; font-weight:800;
           padding:8px 12px; border-radius:9px; font-size:13px; letter-spacing:.5px; }}
  .brand-txt b {{ color:{TEAL}; letter-spacing:2px; font-size:12px; display:block; }}
  .brand-txt span {{ color:{TEXTO_TENUE}; font-size:13px; }}
  .pill-modelo {{ border:1px solid rgba(43,212,196,.4); border-radius:20px; padding:6px 14px;
                  color:{TEAL_TENUE}; font-size:12px; letter-spacing:1px; }}
  .pill-modelo .dot {{ color:{VERDE}; }}
  .hero {{ background:linear-gradient(120deg, rgba(27,143,156,.55), rgba(43,212,196,.18));
           border:1px solid rgba(43,212,196,.25); border-radius:18px; padding:26px 30px; margin:14px 0 22px;
           box-shadow:0 8px 30px rgba(0,0,0,.3); }}
  .hero h1 {{ margin:0; font-size:27px; }}
  .hero p {{ margin:8px 0 0; color:{TEXTO_TENUE}; font-size:14px; }}

  .panel {{ background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.09);
            border-radius:16px; padding:18px; margin:16px 0; box-shadow:0 4px 16px rgba(0,0,0,.2); }}
  .panel h2 {{ margin:2px 4px 4px; font-size:17px; }}
  .panel .hint {{ color:{TEXTO_TENUE}; font-size:12.5px; margin:0 4px 12px; }}

  .filtros {{ display:grid; grid-template-columns: 200px 1fr 1fr 1fr auto; gap:14px; align-items:end; }}
  .filtros label {{ display:block; color:{TEXTO_TENUE}; font-size:11px; letter-spacing:.5px; margin-bottom:5px; }}
  select {{ width:100%; background:#0c2230; color:{TEXTO}; border:1px solid rgba(43,212,196,.25);
            border-radius:9px; padding:10px 12px; font-size:14px; }}
  .btn {{ background:rgba(43,212,196,.12); border:1px solid rgba(43,212,196,.35); color:{TEAL_TENUE};
          border-radius:9px; padding:10px 16px; cursor:pointer; font-size:13px; }}
  .filtro-titulo {{ color:{TEAL}; font-size:13px; letter-spacing:1px; align-self:center; }}

  .kpi-row {{ display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin:16px 0; }}
  .kpi {{ background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); border-radius:15px; padding:18px; }}
  .kpi .lbl {{ color:{TEXTO_TENUE}; font-size:11px; letter-spacing:.6px; text-transform:uppercase; }}
  .kpi .big {{ font-size:38px; font-weight:800; line-height:1.1; margin-top:8px; }}
  .kpi .sub {{ color:{TEXTO_TENUE}; font-size:12px; margin-top:6px; }}
  .barra {{ height:6px; border-radius:4px; background:rgba(255,255,255,.1); margin-top:10px; overflow:hidden; }}
  .barra > i {{ display:block; height:100%; border-radius:4px; }}
  .chip {{ display:inline-block; background:rgba(43,212,196,.12); border:1px solid rgba(43,212,196,.3);
           color:{TEAL_TENUE}; border-radius:8px; padding:5px 10px; font-size:12px; margin:3px 4px 3px 0; }}
  .badge-pend {{ display:inline-block; background:rgba(240,134,62,.15); border:1px solid rgba(240,134,62,.4);
                 color:#f6b27e; border-radius:6px; padding:2px 8px; font-size:10.5px; letter-spacing:.4px; }}
  .badge-modelo {{ display:inline-block; background:rgba(43,212,196,.16); border:1px solid rgba(43,212,196,.5);
                   color:{TEAL_TENUE}; border-radius:6px; padding:2px 9px; font-size:10.5px; letter-spacing:.5px; vertical-align:middle; }}

  .pred-panel {{ background:linear-gradient(120deg, rgba(43,212,196,.10), rgba(27,143,156,.05));
                 border:1px solid rgba(43,212,196,.35); }}
  .pred-grid {{ display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin-top:6px; }}
  .pcard {{ background:rgba(8,24,34,.55); border:1px solid rgba(43,212,196,.18); border-radius:13px; padding:16px 16px 14px; }}
  .pc-ugt {{ color:{TEXTO_TENUE}; font-size:12.5px; font-weight:600; }}
  .pc-val {{ font-size:40px; font-weight:800; line-height:1; margin-top:8px; }}
  .pc-pct {{ font-size:18px; color:{TEXTO_TENUE}; font-weight:600; }}
  .pc-barra {{ height:6px; border-radius:4px; background:rgba(255,255,255,.1); margin-top:10px; overflow:hidden; }}
  .pc-barra > i {{ display:block; height:100%; border-radius:4px; }}
  .pc-nivel {{ font-size:12px; font-weight:700; margin-top:8px; letter-spacing:.4px; }}

  .grid2 {{ display:grid; grid-template-columns: 1.15fr .85fr; gap:16px; }}
  .mapa-grid {{ display:grid; grid-template-columns: repeat(3,1fr); gap:10px; }}
  .dcard {{ background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); border-left-width:4px;
            border-radius:11px; padding:12px 13px; cursor:default; transition:transform .1s; }}
  .dcard:hover {{ transform:translateY(-2px); background:rgba(255,255,255,.06); }}
  .dcard .dn {{ font-weight:700; font-size:13.5px; }}
  .dcard .dp {{ color:{TEXTO_TENUE}; font-size:11px; margin-top:2px; }}
  .dcard .ds {{ float:right; font-weight:800; font-size:15px; border-radius:7px; padding:1px 9px; color:#06141c; }}
  .leyenda {{ display:flex; gap:18px; font-size:12px; color:{TEXTO_TENUE}; margin:2px 4px 12px; }}
  .leyenda b {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px; vertical-align:middle; }}

  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:{TEXTO_TENUE}; font-weight:600; padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.12);
        font-size:11px; letter-spacing:.4px; text-transform:uppercase; }}
  td {{ padding:10px; border-bottom:1px solid rgba(255,255,255,.06); }}
  .score-pill {{ font-weight:800; border-radius:7px; padding:2px 9px; color:#06141c; }}

  .sim {{ display:grid; grid-template-columns: repeat(3,1fr); gap:22px; opacity:.75; }}
  .sim .s-lbl {{ font-weight:600; font-size:13px; }}
  .sim .s-desc {{ color:{TEXTO_TENUE}; font-size:11.5px; margin-top:4px; }}
  input[type=range] {{ width:100%; accent-color:{TEAL}; margin-top:8px; }}

  footer {{ color:{TEXTO_TENUE}; font-size:11.5px; text-align:center; margin-top:26px; line-height:1.7;
            border-top:1px solid rgba(255,255,255,.08); padding-top:18px; }}
  @media (max-width:900px) {{ .grid2,.kpi-row,.mapa-grid,.filtros,.sim,.pred-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <div class="logo">PROTECTA PERÚ</div>
      <div class="brand-txt"><b>SISTEMA DE ALERTA TEMPRANA</b><span>Análisis de conflictos sociales</span></div>
    </div>
    <div class="pill-modelo"><span class="dot">●</span> ÍNDICE DE ACTIVIDAD · ÁNCASH</div>
  </div>

  <div class="hero">
    <h1>Monitor de Conflictos Sociales — Áncash</h1>
    <p>Distritos de interés de ANTAMINA · predicción del modelo + actividad observada · {fecha_ref}</p>
  </div>
{seccion_prediccion}

  <div class="panel">
    <div class="filtros">
      <div class="filtro-titulo">▼ FILTROS JERÁRQUICOS</div>
      <div><label>UNIDAD DE GESTIÓN TERRITORIAL</label><select id="f-ugt"></select></div>
      <div><label>PROVINCIA</label><select id="f-prov"></select></div>
      <div><label>DISTRITO</label><select id="f-dist"></select></div>
      <button class="btn" onclick="reset()">Restablecer</button>
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi">
      <div class="lbl">Índice de actividad regional</div>
      <div class="big" id="kpi-indice" style="color:{color_reg}">{indice_reg}<span style="font-size:18px;color:{TEXTO_TENUE}">/100</span></div>
      <div class="sub" id="kpi-estado">{estado_reg} · promedio de distritos filtrados</div>
      <div class="barra"><i id="kpi-barra" style="width:{indice_reg}%;background:{color_reg}"></i></div>
    </div>
    <div class="kpi">
      <div class="lbl">Distritos en nivel alto</div>
      <div class="big" id="kpi-alerta" style="color:{ROJO}">{n_alerta}</div>
      <div class="sub">de <span id="kpi-total">{len(idx)}</span> · índice ≥ 55 (actividad reciente elevada)</div>
    </div>
    <div class="kpi">
      <div class="lbl">Tipos de conflictividad observada</div>
      <div style="margin-top:12px">{chips}</div>
      <div class="sub" style="margin-top:10px">Drivers socioeconómicos (agua, empleo, gobernanza) <span class="badge-pend">PENDIENTE DE FUENTE</span></div>
    </div>
    <div class="kpi">
      <div class="lbl">Compromiso de las partes</div>
      <div class="big" style="color:{TEXTO_TENUE}">—</div>
      <div class="sub">Requiere registro de mesas de diálogo <span class="badge-pend">PENDIENTE DE FUENTE</span></div>
    </div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Mapa de riesgo por distrito</h2>
      <p class="hint">Índice de actividad observada 0–100 (eventos recientes ponderados). No es una predicción del modelo ML.</p>
      <div class="leyenda">
        <span><b style="background:{VERDE}"></b>Bajo (0–34)</span>
        <span><b style="background:{AMBAR}"></b>Medio (35–54)</span>
        <span><b style="background:{NARANJA}"></b>Alto (55–69)</span>
        <span><b style="background:{ROJO}"></b>Crítico (70+)</span>
      </div>
      <div class="mapa-grid" id="mapa"></div>
    </div>
    <div>
      <div class="panel">
        <h2>Actividad de conflictos en el tiempo</h2>
        <p class="hint">Eventos mensuales reportados en los distritos de interés (dato observado).</p>
        {div_serie}
      </div>
      <div class="panel">
        <h2>Composición de la conflictividad</h2>
        <p class="hint">Distribución real por tipo de evento. Reemplaza el radar de drivers del prototipo (pendiente de fuente).</p>
        {div_radar}
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>Tabla de detalle por distrito</h2>
    <p class="hint">Jerarquía territorial · índice de actividad · acción recomendada (derivada del nivel de actividad).</p>
    <table>
      <thead><tr><th>UGT</th><th>Provincia</th><th>Distrito</th><th>Eventos</th><th>Últ. evento</th><th>Índice</th><th>Acción recomendada</th></tr></thead>
      <tbody id="tabla"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Simulador de escenarios <span class="badge-pend">DEMO — NO CONECTADO A DATOS</span></h2>
    <p class="hint">Vista de la funcionalidad prevista. Requiere un modelo causal con variables de inversión, hídricas y de
       gobernanza que el proyecto aún no incorpora.</p>
    <div class="sim">
      <div><div class="s-lbl">Aumento de inversión minera</div><input type="range" min="0" max="100" value="30" disabled><div class="s-desc">Mayor presión sobre uso de suelo y empleo local.</div></div>
      <div><div class="s-lbl">Sequía / estrés hídrico</div><input type="range" min="0" max="100" value="35" disabled><div class="s-desc">Intensifica conflictos por gestión del agua.</div></div>
      <div><div class="s-lbl">Estabilidad del liderazgo político</div><input type="range" min="0" max="100" value="55" disabled><div class="s-desc">Menor estabilidad debilita la gobernanza local.</div></div>
    </div>
  </div>

  <footer>
    PROTECTA PERÚ · Dashboard de actividad de conflictos — Región Áncash (distritos de interés de ANTAMINA)<br>
    El <b>índice por distrito refleja actividad observada</b> (eventos efectivamente reportados, ponderados por tipo y recencia),
    <b>no una predicción del modelo de Machine Learning</b>. Áncash minero corresponde al Track A, que no cuenta con modelo
    predictivo desplegado. Las secciones marcadas «pendiente de fuente» requieren datos aún no disponibles en el proyecto.
  </footer>
</div>

<script>
const DISTRITOS = {distritos_js};
const TODAS = "— Todas —";

function color(score) {{
  if (score >= 70) return "{ROJO}";
  if (score >= 55) return "{NARANJA}";
  if (score >= 35) return "{AMBAR}";
  return "{VERDE}";
}}
function estadoReg(s) {{
  if (s >= 55) return ["Alto", "{ROJO}"];
  if (s >= 35) return ["Medio", "{AMBAR}"];
  return ["Bajo", "{VERDE}"];
}}

function unicos(campo, filtro) {{
  const vals = DISTRITOS.filter(filtro).map(d => d[campo]);
  return [...new Set(vals)].sort();
}}

function llenarSelect(sel, valores) {{
  sel.innerHTML = "";
  const opTodas = document.createElement("option"); opTodas.textContent = TODAS; sel.appendChild(opTodas);
  valores.forEach(v => {{ const o = document.createElement("option"); o.textContent = v; sel.appendChild(o); }});
}}

function filtroActual() {{
  const ugt = document.getElementById("f-ugt").value;
  const prov = document.getElementById("f-prov").value;
  const dist = document.getElementById("f-dist").value;
  return d => (ugt === TODAS || d.ugt === ugt) && (prov === TODAS || d.provincia === prov) && (dist === TODAS || d.distrito === dist);
}}

function render() {{
  const f = filtroActual();
  const datos = DISTRITOS.filter(f).sort((a,b) => b.score - a.score);

  // KPIs
  const prom = datos.length ? Math.round(datos.reduce((s,d)=>s+d.score,0)/datos.length) : 0;
  const [est, col] = estadoReg(prom);
  document.getElementById("kpi-indice").innerHTML = prom + "<span style='font-size:18px;color:{TEXTO_TENUE}'>/100</span>";
  document.getElementById("kpi-indice").style.color = col;
  document.getElementById("kpi-estado").textContent = est + " · promedio de distritos filtrados";
  document.getElementById("kpi-barra").style.width = prom + "%";
  document.getElementById("kpi-barra").style.background = col;
  document.getElementById("kpi-alerta").textContent = datos.filter(d=>d.score>=55).length;
  document.getElementById("kpi-total").textContent = datos.length;

  // Mapa (tarjetas)
  const mapa = document.getElementById("mapa");
  mapa.innerHTML = datos.map(d => {{
    const c = color(d.score);
    return `<div class="dcard" style="border-left-color:${{c}}">
        <span class="ds" style="background:${{c}}">${{d.score}}</span>
        <div class="dn">${{d.distrito}}</div>
        <div class="dp">${{d.ugt}} · ${{d.provincia}}</div>
      </div>`;
  }}).join("");

  // Tabla
  const tabla = document.getElementById("tabla");
  tabla.innerHTML = datos.map(d => {{
    const c = color(d.score);
    return `<tr>
        <td>${{d.ugt}}</td><td>${{d.provincia}}</td><td><b>${{d.distrito}}</b></td>
        <td>${{d.n_total}}</td><td>${{d.ultimo}}</td>
        <td><span class="score-pill" style="background:${{c}}">${{d.score}}</span></td>
        <td>${{d.accion}}</td>
      </tr>`;
  }}).join("");
}}

function onUgtChange() {{
  const ugt = document.getElementById("f-ugt").value;
  const fProv = d => (ugt === TODAS || d.ugt === ugt);
  llenarSelect(document.getElementById("f-prov"), unicos("provincia", fProv));
  llenarSelect(document.getElementById("f-dist"), unicos("distrito", fProv));
  render();
}}
function onProvChange() {{
  const ugt = document.getElementById("f-ugt").value;
  const prov = document.getElementById("f-prov").value;
  const fDist = d => (ugt === TODAS || d.ugt === ugt) && (prov === TODAS || d.provincia === prov);
  llenarSelect(document.getElementById("f-dist"), unicos("distrito", fDist));
  render();
}}
function reset() {{
  llenarSelect(document.getElementById("f-ugt"), unicos("ugt", () => true));
  llenarSelect(document.getElementById("f-prov"), unicos("provincia", () => true));
  llenarSelect(document.getElementById("f-dist"), unicos("distrito", () => true));
  render();
}}

document.getElementById("f-ugt").addEventListener("change", onUgtChange);
document.getElementById("f-prov").addEventListener("change", onProvChange);
document.getElementById("f-dist").addEventListener("change", render);
reset();
</script>
</body>
</html>"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    idx = calcular_indice()
    serie = serie_mensual()
    comp = composicion_categorias()

    fig_serie = figura_serie(serie)
    fig_radar = figura_composicion(comp)

    pred, meta = _obtener_prediccion()

    html = construir_html(idx, fig_serie, fig_radar, comp, pred, meta)
    SALIDA_HTML.parent.mkdir(parents=True, exist_ok=True)
    SALIDA_HTML.write_text(html, encoding="utf-8")

    print(f"✓ Dashboard de Áncash guardado en {SALIDA_HTML}")
    print(f"  Índice regional: {round(idx['score'].mean())}/100 · Distritos en alto: {(idx['score']>=55).sum()}/{len(idx)}")
    if pred is not None:
        top = pred.iloc[0]
        print(f"  Predicción UGT (30d): mayor riesgo {top['ugt']} {top['probabilidad']*100:.0f}%")

    if "--no-open" not in sys.argv:
        try:
            webbrowser.open(SALIDA_HTML.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
