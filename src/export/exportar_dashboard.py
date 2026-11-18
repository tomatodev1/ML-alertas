"""Exporta el paquete de datos del dashboard de alertas tempranas (Antamina).

Genera `export_dashboard/` con los CSV + modelo_meta.json del contrato del
dashboard estático. Toda la serie de probabilidades sale de UNA especificación
de modelo (src/models/train_ancash.py, MODELO_FINAL):
  - origen = oof_backfill   : semanas con label observable, predichas
                              out-of-fold (walk-forward con purga: cada semana
                              la predice un modelo entrenado solo con datos
                              que existían en ese momento).
  - origen = scoring_modelo : semanas recientes sin label observable,
                              predichas por el modelo final.

Uso (desde la raíz del repo):  python export_dashboard.py [--help]
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from src.dataset import build_ancash
from src.dataset.build_ancash import UGTS
from src.models import evaluacion_ancash as ev
from src.models import recalibrar_umbral
from src.models.train_ancash import MASTER, OOF_PATH, TARGET, entrenar
from src.scoring.ancash_datos import JERARQUIA, _cargar_incidentes_raw, _norm
from src.scoring.score_ancash import cargar_modelo, puntuar

SALIDA_DIR = Path("export_dashboard")
UGT_ID = {
    "Mina San Marcos": "mina_san_marcos",
    "Huallanca": "huallanca",
    "Huarmey": "huarmey",
    "Valle Fortaleza": "valle_fortaleza",
}
DATA_DIR = Path("src/data")
FECHAS_CRITICAS_XLSX = DATA_DIR / "Fechas_Criticas_Antamina.xlsx"
CALENDARIO_CONFLICTOS_XLSX = DATA_DIR / "Calendario_Conflictos_UGT_Antamina.xlsx"
CALENDARIO_ELECTORAL_XLSX = DATA_DIR / "calendario_electoral_2026.xlsx"
HISTORICA_XLSX = DATA_DIR / "dataHistoricaProtestas.xlsx"
PROTESTAS_XLSX = DATA_DIR / "dataProtestas.xlsx"
ACTORES_JSON = Path("data/interim/actores_antamina.json")
REPORTES_PARQUET = Path("data/interim/reportes_antamina_parrafos.parquet")

COLUMNAS = {
    "scoring_modelo.csv": ["ugt", "semana", "prob", "nivel"],
    "scoring_separadas.csv": ["ugt", "semana", "prob", "nivel", "origen"],
    "features_historicas.csv": ["ugt", "semana", "n_eventos_1w", "n_eventos_4w", "dias_desde_ultima_protesta",
                                "racha_semanas_protesta", "rep_antamina_neg_4w", "rep_compromiso_4w"],
    "calibracion_modelo.csv": ["nivel", "umbral", "n", "tasa_acierto", "recall", "descripcion"],
    "calendario_criticidad.csv": ["evento", "tipo", "criticidad", "fecha", "inicio_mes_dia", "ugts",
                                  "evidencia_historica", "fuente_url", "usar_como_proxima_fecha_critica"],
    "alertas_electorales_2026.csv": ["fase", "periodo", "riesgo", "accion_recomendada", "zonas_prioritarias"],
    "eventos_historicos.csv": ["fecha", "ugt", "evento", "tipo", "severidad", "fuente_url"],
    "actores_criticos.csv": ["nombre", "cargo_o_rol", "distrito_o_zona", "n_apariciones", "ultima_mencion",
                             "contexto_resumen"],
}


# ── utilidades ────────────────────────────────────────────────────────────────
def _fecha(s) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def _texto(v, n: int = 400) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", str(v))
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _escribir(df: pd.DataFrame, nombre: str) -> None:
    cols = COLUMNAS[nombre]
    df[cols].to_csv(SALIDA_DIR / nombre, index=False, encoding="utf-8", lineterminator="\n")


def _ugt_ids(texto: str) -> str:
    """'Mina-San Marcos, Huallanca' -> 'mina_san_marcos,huallanca'."""
    t = _norm(texto).replace("-", " ")
    ids = [UGT_ID[u] for u in UGTS if _norm(u) in t]
    return ",".join(ids) if ids else "TODAS"


# ── 1-2. Serie de probabilidades ──────────────────────────────────────────────
def serie_probabilidades(master: pd.DataFrame, oof: pd.DataFrame, paquete: dict, umbrales: dict) -> pd.DataFrame:
    ua, um = umbrales["umbral_alto"], umbrales["umbral_medio"]
    oof = oof.copy()
    oof["origen"] = "oof_backfill"

    vivo = puntuar(master, paquete, umbrales)
    semanas = sorted(vivo["semana_fin"].unique())
    sin_label = set(vivo.loc[vivo[TARGET].isna(), "semana_fin"].unique()) | set(semanas[-2:])
    vivo = vivo[vivo["semana_fin"].isin(sin_label)][["ugt", "semana_fin", "prob"]].copy()
    vivo["origen"] = "scoring_modelo"

    oof = oof[~oof["semana_fin"].isin(vivo["semana_fin"])]
    serie = pd.concat([oof[["ugt", "semana_fin", "prob", "origen"]], vivo], ignore_index=True)
    serie["prob"] = serie["prob"].clip(0, 1).round(4)
    serie["nivel"] = np.where(serie["prob"] >= ua, "ALTO", np.where(serie["prob"] >= um, "MEDIO", "BAJO"))
    serie["semana"] = _fecha(serie["semana_fin"])
    serie["ugt"] = serie["ugt"].map(UGT_ID)
    return serie.sort_values(["semana", "ugt"]).reset_index(drop=True)


# ── 3. Features de contexto ───────────────────────────────────────────────────
def features_historicas(master: pd.DataFrame, desde: str) -> pd.DataFrame:
    f = master[master["semana_fin"] >= pd.Timestamp(desde)].copy()
    out = pd.DataFrame({
        "ugt": f["ugt"].map(UGT_ID),
        "semana": _fecha(f["semana_fin"]),
        "n_eventos_1w": f["inc_prot_1w"].astype(int),
        "n_eventos_4w": f["inc_prot_4w"].astype(int),
        "dias_desde_ultima_protesta": f["dias_desde_ultima_prot"].round().astype("Int64"),
        "racha_semanas_protesta": f["racha_prot"].astype(int),
        "rep_antamina_neg_4w": f["rep_antamina_neg_4w"].astype(int),
        "rep_compromiso_4w": f["rep_compromiso_4w"].astype(int),
    })
    return out.sort_values(["semana", "ugt"]).reset_index(drop=True)


# ── 4. Calibración por nivel ──────────────────────────────────────────────────
def calibracion(oof: pd.DataFrame, umbrales: dict) -> pd.DataFrame:
    ua, um = umbrales["umbral_alto"], umbrales["umbral_medio"]
    t = ev.metricas_por_nivel(oof[TARGET], oof["prob"], ua, um)
    base = float(oof[TARGET].mean())
    desc = {
        "ALTO": (f"prob >= {ua:.2f}. tasa_acierto = % de semanas-UGT en ALTO con protesta/bloqueo en los 30 días "
                 f"siguientes; recall = % de las semanas con protesta que cayeron en ALTO."),
        "MEDIO": (f"{um:.2f} <= prob < {ua:.2f} (MEDIO = tasa base OOF {base:.0%}: riesgo por encima del promedio). "
                  f"tasa_acierto = % con protesta; recall = % de las semanas con protesta que cayeron en MEDIO."),
        "BAJO": (f"prob < {um:.2f}. tasa_acierto = % de semanas-UGT en BAJO SIN protesta en los 30 días siguientes; "
                 f"recall = % de las semanas sin protesta que cayeron en BAJO."),
    }
    t["descripcion"] = t["nivel"].map(desc) + f" OOF walk-forward, n total = {len(oof)}."
    t["umbral"] = t["umbral"].round(2)
    return t


# ── 5. Calendario de criticidad ───────────────────────────────────────────────
def _hipervinculos(ruta: Path) -> dict[tuple[int, int], str]:
    wb = openpyxl.load_workbook(ruta)
    ws = wb.worksheets[0]
    return {(c.row, c.column): c.hyperlink.target for row in ws.iter_rows() for c in row if c.hyperlink}


def _ventanas_electorales() -> pd.DataFrame:
    raw = pd.read_excel(CALENDARIO_CONFLICTOS_XLSX, sheet_name="Ventanas Electorales", header=None)
    fila_enc = next(i for i in range(len(raw)) if str(raw.iloc[i, 0]).strip() == "Periodo operativo")
    df = raw.iloc[fila_enc + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[fila_enc]]
    df = df[df["Fase"].notna()]
    # filtra cualquier repetición del encabezado como dato
    return df[df["Periodo operativo"].astype(str).str.strip() != "Periodo operativo"].reset_index(drop=True)


def _calendario_electoral() -> pd.DataFrame:
    raw = pd.read_excel(CALENDARIO_ELECTORAL_XLSX, header=None)
    links = _hipervinculos(CALENDARIO_ELECTORAL_XLSX)
    fila_enc = next(i for i in range(len(raw)) if str(raw.iloc[i, 0]).strip() == "Fecha")
    filas = []
    for i in range(fila_enc + 1, len(raw)):
        celda = raw.iloc[i, 0]
        # solo celdas con fecha real de Excel; "Dic 2026" / "Mediados nov 2026" no tienen día exacto
        if not isinstance(celda, (pd.Timestamp, datetime.datetime)):
            continue
        fecha = pd.Timestamp(celda)
        filas.append({"fecha": fecha, "hito": str(raw.iloc[i, 1]).strip(), "estado": str(raw.iloc[i, 2]).strip(),
                      # openpyxl es 1-indexado; fila de pandas i (header=None) = fila excel i+1
                      "url": links.get((i + 1, 4), "")})
    return pd.DataFrame(filas)


# Vocabulario que colorea el dashboard (index.html): ALTO / MEDIO-ALTO / MEDIO.
# No tiene nivel superior a ALTO: "Muy alto" y "CRÍTICA" se exportan como ALTO.
CRITICIDAD_DASHBOARD = {"medio": "MEDIO", "media": "MEDIO", "medio-alto": "MEDIO-ALTO", "alto": "ALTO",
                        "alta": "ALTO", "muy alto": "ALTO", "critica": "ALTO"}


def _criticidad(v) -> str:
    t = _norm(_texto(v))
    return CRITICIDAD_DASHBOARD.get(t, _texto(v).upper())


def calendario_criticidad() -> pd.DataFrame:
    filas = []
    fc = pd.read_excel(FECHAS_CRITICAS_XLSX, sheet_name=0)
    for _, r in fc.iterrows():
        filas.append({
            "evento": _texto(r["evento"]), "tipo": _texto(r["tipo"]), "criticidad": _criticidad(r["criticidad"]),
            "fecha": "", "inicio_mes_dia": _texto(r["inicio_mes_dia"]), "ugts": _ugt_ids(str(r["ugts"])),
            "evidencia_historica": _texto(r["evidencia_historica"], 600), "fuente_url": _texto(r["fuente_url"], 1000),
            # la hoja marca en `fecha` "NO CALIFICA" cuando no cumple el criterio interanual
            "usar_como_proxima_fecha_critica": "false" if "NO CALIFICA" in str(r["fecha"]).upper() else "true",
        })

    # Jornadas electorales: fecha exacta, del calendario electoral provisto (nov-2026)
    ventanas = _ventanas_electorales()
    cal = _calendario_electoral()
    hoy = pd.Timestamp.today().normalize()
    jornadas = [
        ("Primera vuelta Elecciones Regionales", "Jornada electoral"),
        ("SEGUNDA VUELTA REGIONAL", "Nueva campaña y jornada"),
    ]
    for patron, fase in jornadas:
        c = cal[cal["hito"].str.contains(patron, case=False, regex=False) & (cal["estado"] != "Reemplazado")]
        v = ventanas[ventanas["Fase"].astype(str).str.strip() == fase]
        if c.empty or v.empty:
            continue
        c, v = c.iloc[0], v.iloc[0]
        filas.append({
            "evento": _texto(c["hito"]).replace("SEGUNDA VUELTA REGIONAL", "Segunda vuelta regional"),
            "tipo": "electoral",
            "criticidad": _criticidad(v["Riesgo"]),
            "fecha": c["fecha"].strftime("%Y-%m-%d"), "inicio_mes_dia": "", "ugts": "TODAS",
            "evidencia_historica": _texto(f"{v['Base de evidencia']}. Estado: {c['estado']}. "
                                          f"Detonantes: {v['Detonantes']}", 600),
            "fuente_url": c["url"],
            "usar_como_proxima_fecha_critica": "true" if c["fecha"] >= hoy else "false",
        })
    return pd.DataFrame(filas)


# ── 6. Alertas electorales ────────────────────────────────────────────────────
def alertas_electorales() -> pd.DataFrame:
    v = _ventanas_electorales()
    cal = _calendario_electoral()
    periodo = v["Periodo operativo"].astype(str).str.strip()
    # La fila "Segunda elección regional, si corresponde" ya tiene fecha oficial:
    # desde la publicación del DS que la convoca hasta la jornada.
    ds = cal[cal["hito"].str.contains("convoca segunda vuelta", case=False)]
    jornada = cal[cal["hito"].str.contains("SEGUNDA VUELTA REGIONAL", regex=False) & (cal["estado"] != "Reemplazado")]
    if not ds.empty and not jornada.empty:
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                 "octubre", "noviembre", "diciembre"]
        a, b = ds["fecha"].iloc[0], jornada["fecha"].iloc[0]
        texto = f"{a.day} {meses[a.month - 1]}–{b.day} {meses[b.month - 1]} {b.year}"
        periodo = periodo.where(~periodo.str.startswith("Segunda elección regional"), texto)
    return pd.DataFrame({
        "fase": v["Fase"].map(_texto), "periodo": periodo, "riesgo": v["Riesgo"].map(_texto),
        "accion_recomendada": v["Acción recomendada"].map(_texto),
        "zonas_prioritarias": v["Zonas prioritarias"].map(_texto),
    })


# ── 7. Eventos históricos ─────────────────────────────────────────────────────
MOTIVOS_PROTESTA = {"PROTESTA", "PROTESTA / PESCA", "POSIBLE PROTESTA", "BLOQUEO", "RECLAMO",
                    "Protestas, Paros y Bloqueos"}
# Vocabulario del dashboard (index.html, severidadColor): ALTO / MEDIO / BAJO
SEVERIDAD = {"alto": "ALTO", "medio": "MEDIO", "bajo": "BAJO",
             "impacto alto": "ALTO", "impacto medio": "MEDIO", "impacto bajo": "BAJO"}


def _tipo_evento(motivo: str, titulo: str) -> str:
    # La categoría genérica del formulario ("Protestas, Paros y Bloqueos") no
    # informa el tipo: se clasifica solo por el titular.
    motivo = "" if str(motivo).startswith("Protestas, Paros") else motivo
    t = _norm(f"{motivo} {titulo}")
    if "bloque" in t:
        return "BLOQUEO"
    if re.search(r"\bparo\b", t):
        return "PARO"
    if "planton" in t:
        return "PLANTÓN"
    if "marcha" in t or "moviliz" in t:
        return "MARCHA"
    return "PROTESTA"


def eventos_historicos() -> pd.DataFrame:
    d2u = {_norm(d): u for u, _, d in JERARQUIA}
    inc = _cargar_incidentes_raw()
    inc["ugt"] = inc["distrito_n"].map(d2u)
    inc = inc[inc["ugt"].notna() & inc["motivo"].isin(MOTIVOS_PROTESTA)].copy()

    # Severidad: relevancia del formulario (impacto sobre la empresa) o la
    # calificación de impacto de dataProtestas para el mismo distrito y fecha.
    sev = pd.Series("", index=inc.index)
    if "relevancia" in inc.columns:
        sev = inc["relevancia"].fillna("").map(lambda r: SEVERIDAD.get(_norm(r), ""))
    if PROTESTAS_XLSX.exists():
        dp = pd.read_excel(PROTESTAS_XLSX, sheet_name=0)
        dp["k"] = list(zip(pd.to_datetime(dp["ID_FECHA"], errors="coerce").dt.normalize(),
                           dp["DISTRITO"].astype(str).map(_norm)))
        cal_col = next(c for c in dp.columns if _norm(c).startswith("calificacion"))
        mapa = dict(zip(dp["k"], dp[cal_col].astype(str).map(lambda r: SEVERIDAD.get(_norm(r), ""))))
        k = list(zip(inc["fecha"].dt.normalize(), inc["distrito_n"]))
        sev = sev.where(sev != "", pd.Series([mapa.get(x, "") for x in k], index=inc.index))

    ev_rec = pd.DataFrame({
        "fecha": _fecha(inc["fecha"]), "ugt": inc["ugt"].map(UGT_ID),
        "evento": inc["titulo"].map(_texto).replace({"nan": ""}),
        "tipo": [_tipo_evento(m, t) for m, t in zip(inc["motivo"], inc["titulo"])],
        "severidad": sev.values, "fuente_url": inc["url"].fillna("").values,
    })
    ev_rec = ev_rec[ev_rec["evento"] != ""]

    # Historial de prensa (2001–), dentro de las 4 UGTs. Sin URL en la fuente.
    filas = []
    if HISTORICA_XLSX.exists():
        h = pd.read_excel(HISTORICA_XLSX, sheet_name="Hoja1")
        h["f"] = pd.to_datetime(h["Fecha_Protesta"], errors="coerce")
        h["ugt"] = h["Distrito"].astype(str).map(_norm).map(d2u)
        h = h[h["f"].notna() & h["ugt"].notna()]
        for _, r in h.iterrows():
            accion = _texto(r.get("Acción_1"))
            texto = (f"{accion or 'Protesta'} en {_texto(r['Distrito'])} — reclamo: {_texto(r.get('Reclamo')) or 's/d'}; "
                     f"adversario: {_texto(r.get('Adversario')) or 's/d'} (prensa: {_texto(r.get('Periódico'))})")
            filas.append({"fecha": r["f"].strftime("%Y-%m-%d"), "ugt": UGT_ID[r["ugt"]], "evento": _texto(texto),
                          "tipo": _tipo_evento(accion, ""), "severidad": "", "fuente_url": ""})
    out = pd.concat([ev_rec, pd.DataFrame(filas)], ignore_index=True)
    out = out.drop_duplicates(subset=["fecha", "ugt", "evento"])
    return out.sort_values(["fecha", "ugt"]).reset_index(drop=True)


# ── 8. Actores críticos ───────────────────────────────────────────────────────
def actores_criticos() -> pd.DataFrame:
    """Actualización incremental del catálogo curado (data/interim/actores_antamina.json):
    a `n_apariciones` se le suman las menciones POSTERIORES a su `ultima_mencion`
    en los reportes de Antamina y en los titulares del formulario de monitoreo;
    `ultima_mencion` y `contexto_resumen` pasan a la mención más reciente.
    No se agregan actores nuevos automáticamente (requiere revisión humana)."""
    catalogo = json.loads(ACTORES_JSON.read_text(encoding="utf-8"))
    corpus = []
    if REPORTES_PARQUET.exists():
        r = pd.read_parquet(REPORTES_PARQUET)
        if "texto" in r.columns:
            r = r.drop_duplicates(subset=["archivo", "texto"])
            corpus.append(pd.DataFrame({"fecha": pd.to_datetime(r["fecha_reporte"]), "texto": r["texto"],
                                        "doc": r["archivo"]}))
    form = pd.read_excel(DATA_DIR / "dataFormulario.xlsx", sheet_name="Hoja1")
    col_t = next(c for c in ["Id_Noticias", "Insertar noticia (Formato tabla de ChatGTP)"] if c in form.columns)
    corpus.append(pd.DataFrame({"fecha": pd.to_datetime(form["Marca temporal"], errors="coerce").dt.normalize(),
                                "texto": form[col_t].astype(str), "doc": "form_" + form.index.astype(str)}))
    corpus = pd.concat(corpus, ignore_index=True).dropna(subset=["fecha"])
    corpus["tn"] = corpus["texto"].map(_norm)

    filas = []
    for a in catalogo:
        ult = pd.Timestamp(a["ultima_mencion"])
        patron = r"\b" + re.escape(_norm(a["nombre"])) + r"\b"
        hits = corpus[(corpus["fecha"] > ult) & corpus["tn"].str.contains(patron, regex=True)]
        hits = hits.drop_duplicates(subset=["doc"]).sort_values("fecha")
        n = int(a["n_apariciones"]) + int(hits["doc"].nunique())
        if not hits.empty:
            ult_txt = hits.iloc[-1]
            ultima, contexto = ult_txt["fecha"].strftime("%Y-%m-%d"), _texto(ult_txt["texto"], 400)
        else:
            ultima, contexto = ult.strftime("%Y-%m-%d"), _texto(a["contexto_resumen"], 400)
        filas.append({"nombre": a["nombre"], "cargo_o_rol": _texto(a["cargo_o_rol"]),
                      "distrito_o_zona": _texto(a["distrito_o_zona"]), "n_apariciones": n,
                      "ultima_mencion": ultima, "contexto_resumen": contexto})
    return pd.DataFrame(filas).sort_values(["ultima_mencion", "n_apariciones"], ascending=False).reset_index(drop=True)


# ── 9. Metadatos ──────────────────────────────────────────────────────────────
def modelo_meta(paquete: dict, umbrales: dict, serie: pd.DataFrame) -> dict:
    m = paquete["metricas_oof"]
    return {
        "modelo": paquete["modelo_tipo"],
        "version": str(pd.Timestamp.today().date()),
        "horizonte_dias": int(paquete["horizonte_dias"]),
        "target": paquete["target"],
        "umbral_alto": float(umbrales["umbral_alto"]),
        "umbral_medio": float(umbrales["umbral_medio"]),
        "n_niveles": 3,
        "features": list(paquete["feature_cols"]),
        "metricas_oof": {"auc": round(m["roc_auc"], 4), "brier": round(m["brier"], 4), "pr_auc": round(m["pr_auc"], 4)},
        "semana_mas_reciente": serie["semana"].max(),
        "entrenado_hasta": paquete["entrenado_hasta"],
        "fecha_corte_datos": paquete["fecha_corte_datos"],
        "metricas_oof_baseline": {"auc": round(paquete["metricas_oof_baseline"]["roc_auc"], 4),
                                  "brier": round(paquete["metricas_oof_baseline"]["brier"], 4),
                                  "pr_auc": round(paquete["metricas_oof_baseline"]["pr_auc"], 4)},
    }


# ── Orquestación ──────────────────────────────────────────────────────────────
def _actualizar_fuentes(con_red: bool) -> None:
    from src.data import loader_calendario, loader_reportes
    loader_calendario.main()
    loader_reportes.main()
    if con_red:
        from src.data import scraper_defensoria
        scraper_defensoria.main()


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Regenera export_dashboard/ (modelo de conflictos Áncash/Antamina).")
    ap.add_argument("--actualizar-fuentes", action="store_true",
                    help="re-parsea reportes (src/data/2025, 2026) y regenera el calendario antes de construir")
    ap.add_argument("--con-red", action="store_true", help="con --actualizar-fuentes: también re-descarga Defensoría")
    ap.add_argument("--sin-reentrenar", action="store_true",
                    help="usa el modelo y el OOF ya guardados (no reentrena)")
    ap.add_argument("--recalibrar", action="store_true",
                    help="recalcula los umbrales ALTO/MEDIO sobre el OOF (cambia los textos del dashboard)")
    args = ap.parse_args(argv)

    if args.actualizar_fuentes:
        _actualizar_fuentes(args.con_red)

    print("1/4 Construyendo tabla maestra UGT × semana…")
    master = build_ancash.construir()
    master.to_parquet(MASTER, index=False)
    print(f"    corte de datos {build_ancash.FECHA_CORTE.date()}, última semana {master['semana_fin'].max().date()}")

    if not args.sin_reentrenar:
        print("2/4 Reentrenando (walk-forward con purga)…")
        try:
            entrenar(master, verbose=True)
        except SystemExit as e:
            print(f"    ⚠ {e} — se mantiene el modelo guardado.")
    else:
        print("2/4 Sin reentrenar: se usa el modelo guardado.")

    paquete = cargar_modelo()
    oof = pd.read_parquet(OOF_PATH)
    if args.recalibrar or not recalibrar_umbral.UMBRALES_PATH.exists():
        u = recalibrar_umbral.calibrar(oof[TARGET].astype(int), oof["prob"])
        recalibrar_umbral.guardar(u, len(oof))
        print(f"    umbrales recalibrados: ALTO {u['umbral_alto']:.2f} · MEDIO {u['umbral_medio']:.2f}")
    umbrales = recalibrar_umbral.cargar()

    print("3/4 Exportando…")
    SALIDA_DIR.mkdir(exist_ok=True)
    serie = serie_probabilidades(master, oof, paquete, umbrales)
    _escribir(serie[serie["origen"] == "scoring_modelo"], "scoring_modelo.csv")
    _escribir(serie, "scoring_separadas.csv")
    _escribir(features_historicas(master, serie["semana"].min()), "features_historicas.csv")
    _escribir(calibracion(oof, umbrales), "calibracion_modelo.csv")
    _escribir(calendario_criticidad(), "calendario_criticidad.csv")
    _escribir(alertas_electorales(), "alertas_electorales_2026.csv")
    _escribir(eventos_historicos(), "eventos_historicos.csv")
    _escribir(actores_criticos(), "actores_criticos.csv")
    (SALIDA_DIR / "modelo_meta.json").write_text(
        json.dumps(modelo_meta(paquete, umbrales, serie), ensure_ascii=False, indent=2), encoding="utf-8")

    print("4/4 Validando contrato…")
    from src.export.validar_dashboard import validar
    errores = validar(SALIDA_DIR)
    if errores:
        print("✗ El paquete NO cumple el contrato:")
        for e in errores:
            print("   -", e)
        raise SystemExit(1)
    ult = serie[serie["semana"] == serie["semana"].max()]
    print(f"✓ Paquete válido en {SALIDA_DIR.resolve()}")
    print(f"  Semana más reciente {ult['semana'].iloc[0]}:")
    for _, r in ult.sort_values("prob", ascending=False).iterrows():
        print(f"    {r['ugt']:<16} {r['prob']:.3f}  {r['nivel']}")


if __name__ == "__main__":
    main()
