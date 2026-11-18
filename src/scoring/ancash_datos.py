"""Capa de datos del dashboard de Áncash (ANTAMINA).

Calcula, para los 18 distritos de interés, un ÍNDICE DE ACTIVIDAD OBSERVADA
(0–100) a partir de la base interna de incidentes geolocalizados. Es un dato
real (eventos efectivamente reportados), con ponderación por tipo y decaimiento
temporal — NO es una predicción del modelo ML (el modelo es departamental y
Áncash minero es Track A, sin modelo desplegable). Esa distinción se mantiene
explícita en todo el dashboard.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVO_INCIDENTES = Path("src/data") / "dataIncidentes.xlsx"
ARCHIVO_FORMULARIO = Path("src/data") / "dataFormulario.xlsx"
# Versiones previas congeladas. La actualización del 2026-11-18 trajo un
# dataIncidentes filtrado (solo accidente/protesta/homicidio) que omite
# registros que la versión de junio sí tenía (robos, detenciones, reclamos).
# Se UNEN ambas versiones y se deduplica, para no perder eventos históricos.
RESPALDOS_INCIDENTES = sorted((Path("src/data") / "respaldo_2026-06").glob("dataIncidentes*.xlsx"))
HOJA = "Hoja1"


def _leer_excel_con_encabezado(ruta: Path, columna_clave: str) -> pd.DataFrame:
    """Lee la primera hoja que contenga `columna_clave` en alguna de sus
    primeras 10 filas, usando esa fila como encabezado (las exportaciones
    nuevas traen filas/columnas vacías antes de la tabla)."""
    hojas = pd.read_excel(ruta, sheet_name=None, header=None, nrows=10)
    for nombre, muestra in hojas.items():
        for i in range(len(muestra)):
            valores = [str(v).strip() for v in muestra.iloc[i].tolist()]
            if columna_clave in valores:
                df = pd.read_excel(ruta, sheet_name=nombre, header=i)
                df.columns = [str(c).strip() for c in df.columns]
                return df.loc[:, ~df.columns.str.startswith("Unnamed")]
    raise ValueError(f"{ruta}: no se encontró la columna {columna_clave!r} en ninguna hoja")

# ── Jerarquía territorial de ANTAMINA (tabla UGT → Provincia → Distrito) ──────
JERARQUIA: list[tuple[str, str, str]] = [
    ("Mina San Marcos", "Huari",          "San Marcos"),
    ("Mina San Marcos", "Huari",          "Chavín de Huántar"),
    ("Mina San Marcos", "Huari",          "Huachis"),
    ("Mina San Marcos", "Huari",          "San Pedro de Chana"),
    ("Huallanca",       "Bolognesi",      "Huallanca"),
    ("Huallanca",       "Bolognesi",      "Aquía"),
    ("Huallanca",       "Bolognesi",      "Chiquián"),
    ("Valle Fortaleza", "Bolognesi",      "Cajacay"),
    ("Valle Fortaleza", "Bolognesi",      "Antonio Raimondi"),
    ("Valle Fortaleza", "Bolognesi",      "Colquioc"),
    ("Valle Fortaleza", "Bolognesi",      "Huayllacayán"),
    ("Valle Fortaleza", "Recuay",         "Catac"),
    ("Valle Fortaleza", "Recuay",         "Pampas Chico"),
    ("Valle Fortaleza", "Recuay",         "Marca"),
    ("Valle Fortaleza", "Recuay",         "Llacllín"),
    ("Valle Fortaleza", "Recuay",         "Pararín"),
    ("Valle Fortaleza", "Barranca - Lima", "Paramonga"),
    ("Huarmey",         "Huarmey",        "Huarmey"),
]

# Departamentos donde pueden estar estos distritos (evita colisión de nombres
# con distritos homónimos de otras regiones, ej. San Marcos de Cajamarca).
DEPTOS_VALIDOS = {"ANCASH", "LIMA"}

# Clasificación de MOTIVO → categoría de conflictividad (mismo criterio que
# loader_incidentes.py). Accidentes/desastres quedan fuera (no son conflicto).
MOTIVO_A_CATEGORIA = {
    "PROTESTA": "PROTESTA", "PROTESTA / PESCA": "PROTESTA", "POSIBLE PROTESTA": "PROTESTA",
    "BLOQUEO": "PROTESTA", "RECLAMO": "PROTESTA", "REUNIÓN": "PROTESTA",
    "EVENTO SOCIAL": "PROTESTA", "SOCIAL": "PROTESTA", "INCIDENTE FUNDO": "PROTESTA",
    "INCIDENTE MINA": "PROTESTA",
    "MOVIMIENTO POLITICO": "POLITICA", "MOVIMIENTO POLÍTICO": "POLITICA", "POLITICO ELECTORAL": "POLITICA",
    "HOMICIDIO": "VIOLENCIA", "SICARIATO": "VIOLENCIA", "SECUESTRO": "VIOLENCIA",
    "EXTORSIÓN": "VIOLENCIA", "EXTORCION": "VIOLENCIA", "BANDA CRIMINAL": "VIOLENCIA",
    "VIOLENCIA CIUDADANA": "VIOLENCIA", "DETENCIÓN": "VIOLENCIA", "ROBO": "VIOLENCIA",
}

# Pesos del índice de actividad: la conflictividad social pesa más que la
# inseguridad común para el propósito de alerta de conflictos.
PESOS = {"PROTESTA": 1.0, "POLITICA": 0.8, "VIOLENCIA": 0.45}

TAU_DIAS = 180  # vida media del decaimiento temporal (~6 meses)

# Umbrales del índice → estado y acción recomendada (réplica del PDF).
def estado_y_accion(score: float) -> tuple[str, str, str]:
    if score >= 70:
        return "Alto", "#e74d6b", "Mesa de diálogo urgente"
    if score >= 55:
        return "Alto", "#f0863e", "Intervención preventiva"
    if score >= 35:
        return "Medio", "#f4c430", "Monitoreo cercano"
    return "Bajo", "#37c98a", "Vigilancia rutinaria"


def _norm(s) -> str:
    s = str(s).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# ── Carga de incidentes (con DISTRITO, sin colapsar a zona) ──────────────────
def _cargar_incidentes_raw() -> pd.DataFrame:
    """Eventos geolocalizados (dataIncidentes.xlsx) + protestas/paros/bloqueos
    de dataFormulario.xlsx (monitoreo de noticias específico de Antamina).

    dataFormulario tiene mucha más densidad de eventos de protesta que
    dataIncidentes, pero es un log de NOTICIAS (varias filas pueden describir
    el mismo evento real) — se deduplica a 1 fila por (distrito, fecha) en
    `_cargar_formulario_protestas_raw()`, y aquí solo se agregan las
    combinaciones (distrito, fecha) que dataIncidentes NO tiene, para no
    contar el mismo evento dos veces si aparece en ambas fuentes.
    """
    partes = [_leer_excel_con_encabezado(r, "ID_FECHA") for r in [ARCHIVO_INCIDENTES, *RESPALDOS_INCIDENTES]]
    raw = pd.concat(partes, ignore_index=True)
    df = pd.DataFrame()
    df["fecha"] = pd.to_datetime(raw["ID_FECHA"], errors="coerce", format="mixed").dt.normalize()
    df["depto_n"] = raw["ID_DEPARTAMENTO"].apply(_norm).str.upper()
    df["distrito_n"] = raw["DISTRITO"].apply(_norm)
    df["motivo"] = raw["MOTIVO"].astype(str).str.strip()
    df["categoria"] = raw["MOTIVO"].apply(lambda m: MOTIVO_A_CATEGORIA.get(_norm(m).upper(), "OTRO"))
    df["titulo"] = raw["TITULO"].astype(str).str.strip()
    df["url"] = raw["LINK"].astype(str).str.strip() if "LINK" in raw.columns else ""
    df["url"] = df["url"].where(df["url"].str.startswith("http"), "")
    df["fuente"] = "dataIncidentes"
    df["lat"] = np.nan
    df["lon"] = np.nan
    df = df[df["fecha"].notna()].copy()
    df = df[df["depto_n"].isin({_norm(d).upper() for d in DEPTOS_VALIDOS})]
    # Dedup entre versiones: misma fecha, distrito y motivo/título normalizados.
    # Se prefiere la fila que trae URL (versión nueva).
    df["_tit_n"] = df["titulo"].apply(_norm)
    df = (df.sort_values("url", ascending=False)
            .drop_duplicates(subset=["fecha", "distrito_n", "categoria", "_tit_n"])
            .drop(columns="_tit_n")
            .sort_values("fecha").reset_index(drop=True))

    formulario = _cargar_formulario_protestas_raw()
    ya_cubiertas = set(zip(
        df.loc[df["categoria"] == "PROTESTA", "distrito_n"],
        df.loc[df["categoria"] == "PROTESTA", "fecha"].dt.normalize(),
    ))
    formulario = formulario[
        ~formulario.apply(lambda r: (r["distrito_n"], r["fecha"]) in ya_cubiertas, axis=1)
    ]
    return pd.concat([df, formulario], ignore_index=True)


def _cargar_formulario_protestas_raw() -> pd.DataFrame:
    """Eventos de categoría 'Protestas, Paros y Bloqueos' de dataFormulario.xlsx,
    deduplicados a 1 fila por (distrito, fecha) — el formulario es un log de
    noticias y un mismo evento real suele generar varias filas (distintos
    artículos sobre el mismo hecho)."""
    raw = pd.read_excel(ARCHIVO_FORMULARIO, sheet_name="Hoja1")
    df = pd.DataFrame()
    df["fecha"] = pd.to_datetime(raw["Marca temporal"], errors="coerce").dt.normalize()
    df["depto_n"] = raw["ID_Departamento"].apply(_norm).str.upper()
    df["distrito_n"] = raw["Distrito"].astype(str).apply(_norm)
    df["categoria"] = "PROTESTA"
    df["motivo"] = "Protestas, Paros y Bloqueos"
    # El formulario cambió de columnas en nov-2026: el titular pasó a
    # `Id_Noticias` y se agregó `Link de la noticia`.
    col_titulo = next((c for c in ["Insertar noticia (Formato tabla de ChatGTP)", "Id_Noticias"] if c in raw.columns), None)
    df["titulo"] = raw[col_titulo].astype(str).str.strip() if col_titulo else ""
    col_link = next((c for c in raw.columns if str(c).startswith("Link")), None)
    df["url"] = raw[col_link].astype(str).str.strip() if col_link else ""
    df["url"] = df["url"].where(df["url"].str.startswith("http"), "")
    col_rel = next((c for c in raw.columns if str(c).startswith("Relevancia")), None)
    df["relevancia"] = raw[col_rel].astype(str).str.strip() if col_rel else ""
    df["fuente"] = "dataFormulario"
    df["lat"] = np.nan
    df["lon"] = np.nan

    es_protesta = raw["Categoria"].astype(str).str.contains("Bloqueo|Paro|Protesta", case=False, na=False)
    df = df[es_protesta & df["fecha"].notna()].copy()
    df = df[df["depto_n"].isin({_norm(d).upper() for d in DEPTOS_VALIDOS})]
    df = df.sort_values("fecha").drop_duplicates(subset=["distrito_n", "fecha"], keep="first")
    return df


# ── Índice de actividad por distrito ──────────────────────────────────────────
def calcular_indice(hoy: pd.Timestamp | None = None) -> pd.DataFrame:
    hoy = hoy or pd.Timestamp.today().normalize()
    inc = _cargar_incidentes_raw()
    inc = inc[inc["categoria"] != "OTRO"].copy()

    # Peso de cada evento = peso_categoría × decaimiento temporal
    inc["peso_cat"] = inc["categoria"].map(PESOS).fillna(0.0)
    dias_atras = (hoy - inc["fecha"]).dt.days.clip(lower=0)
    inc["peso"] = inc["peso_cat"] * np.exp(-dias_atras / TAU_DIAS)

    filas = []
    for ugt, provincia, distrito in JERARQUIA:
        dn = _norm(distrito)
        sub = inc[inc["distrito_n"] == dn]
        score_raw = float(sub["peso"].sum())
        n_total = int(len(sub))
        n_prot = int((sub["categoria"] == "PROTESTA").sum())
        n_viol = int((sub["categoria"] == "VIOLENCIA").sum())
        n_pol = int((sub["categoria"] == "POLITICA").sum())
        ultimo = sub["fecha"].max()
        filas.append({
            "ugt": ugt, "provincia": provincia, "distrito": distrito,
            "score_raw": score_raw, "n_total": n_total,
            "n_protesta": n_prot, "n_violencia": n_viol, "n_politica": n_pol,
            "ultimo_evento": ultimo,
        })

    df = pd.DataFrame(filas)

    # Normalización 0–100: relativa al percentil 95 (evita que un outlier
    # defina el tope), comprimida con raíz para repartir mejor el rango medio.
    ref = np.percentile(df["score_raw"][df["score_raw"] > 0], 95) if (df["score_raw"] > 0).any() else 1.0
    ref = max(ref, 1e-9)
    df["score"] = (100 * np.sqrt((df["score_raw"] / ref).clip(upper=1.0))).round().astype(int)
    # Distritos con eventos pero score 0 por redondeo → mínimo 1 si hubo actividad
    df.loc[(df["n_total"] > 0) & (df["score"] == 0), "score"] = 1

    estados = df["score"].apply(estado_y_accion)
    df["estado"] = [e[0] for e in estados]
    df["color"] = [e[1] for e in estados]
    df["accion"] = [e[2] for e in estados]

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── Serie temporal mensual (actividad regional observada) ─────────────────────
def serie_mensual(meses: int = 24) -> pd.DataFrame:
    inc = _cargar_incidentes_raw()
    inc = inc[inc["categoria"] != "OTRO"].copy()
    distritos_n = {_norm(d) for _, _, d in JERARQUIA}
    inc = inc[inc["distrito_n"].isin(distritos_n)]

    inc["mes"] = inc["fecha"].dt.to_period("M").dt.to_timestamp()
    serie = inc.groupby("mes").size().rename("eventos").reset_index()

    # Completar meses faltantes en el rango
    if not serie.empty:
        rango = pd.date_range(serie["mes"].min(), serie["mes"].max(), freq="MS")
        serie = serie.set_index("mes").reindex(rango, fill_value=0).rename_axis("mes").reset_index()
        serie = serie.tail(meses)
    return serie


# ── Composición por tipo de conflictividad (para el radar real) ──────────────
def composicion_categorias() -> pd.DataFrame:
    df = calcular_indice()
    total = {
        "Protesta / social": int(df["n_protesta"].sum()),
        "Inseguridad / violencia": int(df["n_violencia"].sum()),
        "Político / electoral": int(df["n_politica"].sum()),
    }
    return pd.Series(total).rename("eventos").reset_index().rename(columns={"index": "categoria"})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    idx = calcular_indice()
    print("ÍNDICE DE ACTIVIDAD POR DISTRITO (18 de interés ANTAMINA):\n")
    print(idx[["ugt", "provincia", "distrito", "n_total", "n_protesta", "score", "estado", "accion"]].to_string(index=False))
    print(f"\nÍndice regional (promedio): {idx['score'].mean():.0f}/100")
    print(f"Distritos con índice ≥ 55: {(idx['score'] >= 55).sum()}/18")
    print(f"\nSerie mensual (últimos meses): {len(serie_mensual())} meses, "
          f"{serie_mensual()['eventos'].sum()} eventos totales")
    print("\nComposición por categoría:")
    print(composicion_categorias().to_string(index=False))
