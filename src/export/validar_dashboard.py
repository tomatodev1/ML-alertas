"""Validador del contrato de `export_dashboard/` (dashboard Antamina).

Uso:  python -m src.export.validar_dashboard [carpeta]
Sale con código 1 si hay errores. Comprueba: archivos, encoding UTF-8,
encabezado único y exacto, IDs de UGT, rangos de prob, formato de fechas,
duplicados ugt+semana, coherencia nivel↔prob↔umbrales y metadatos.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.export.exportar_dashboard import COLUMNAS

UGT_IDS = {"mina_san_marcos", "huallanca", "huarmey", "valle_fortaleza"}
NIVELES = {"ALTO", "MEDIO", "BAJO"}
ORIGENES = {"oof_backfill", "scoring_modelo"}
RE_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RE_MMDD = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
META_CLAVES = {"modelo", "version", "horizonte_dias", "target", "umbral_alto", "umbral_medio", "n_niveles",
               "features", "metricas_oof", "semana_mas_reciente", "entrenado_hasta"}


def _fechas_validas(s: pd.Series) -> bool:
    s = s.astype(str)
    return bool(s.str.match(RE_FECHA).all() and pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").notna().all())


def _leer(ruta: Path, errores: list[str]) -> pd.DataFrame | None:
    nombre = ruta.name
    try:
        crudo = ruta.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        errores.append(f"{nombre}: no es UTF-8 válido")
        return None
    if crudo.startswith("﻿"):
        errores.append(f"{nombre}: tiene BOM (el encabezado se leería como '\\ufeff{COLUMNAS[nombre][0]}')")
    filas = list(csv.reader(io.StringIO(crudo.lstrip("﻿"))))
    if not filas:
        errores.append(f"{nombre}: vacío")
        return None
    if filas[0] != COLUMNAS[nombre]:
        errores.append(f"{nombre}: encabezado {filas[0]} ≠ contrato {COLUMNAS[nombre]}")
        return None
    if any(f == COLUMNAS[nombre] for f in filas[1:]):
        errores.append(f"{nombre}: la fila de encabezado se repite como dato")
    if len(filas) < 2:
        errores.append(f"{nombre}: sin filas de datos")
    largos = {len(f) for f in filas}
    if len(largos) != 1:
        errores.append(f"{nombre}: filas con distinto número de columnas {sorted(largos)}")
    return pd.read_csv(io.StringIO(crudo.lstrip("﻿")), dtype=str, keep_default_na=False)


def _check_serie(df: pd.DataFrame, nombre: str, meta: dict, errores: list[str]) -> None:
    if not df["ugt"].isin(UGT_IDS).all():
        errores.append(f"{nombre}: ugt fuera de los 4 IDs: {sorted(set(df['ugt']) - UGT_IDS)}")
    if not _fechas_validas(df["semana"]):
        errores.append(f"{nombre}: 'semana' con formato distinto de YYYY-MM-DD")
    elif (pd.to_datetime(df["semana"]).dt.weekday != 6).any():
        errores.append(f"{nombre}: hay semanas que no cierran en domingo")
    if df.duplicated(["ugt", "semana"]).any():
        errores.append(f"{nombre}: {int(df.duplicated(['ugt', 'semana']).sum())} duplicados ugt+semana")
    if "prob" in df:
        if not df["prob"].str.match(r"^(0(\.\d+)?|1(\.0+)?)$").all():
            errores.append(f"{nombre}: 'prob' no numérica en [0,1] con punto decimal")
        else:
            p = df["prob"].astype(float)
            esperado = pd.Series("BAJO", index=df.index)
            esperado[p >= meta["umbral_medio"]] = "MEDIO"
            esperado[p >= meta["umbral_alto"]] = "ALTO"
            malos = (df["nivel"] != esperado).sum()
            if malos:
                errores.append(f"{nombre}: {int(malos)} filas con 'nivel' inconsistente con prob y umbrales del meta")
    if "nivel" in df and not df["nivel"].isin(NIVELES).all():
        errores.append(f"{nombre}: niveles inválidos {sorted(set(df['nivel']) - NIVELES)}")


def validar(carpeta: Path | str = "export_dashboard") -> list[str]:
    carpeta = Path(carpeta)
    errores: list[str] = []
    ruta_meta = carpeta / "modelo_meta.json"
    if not ruta_meta.exists():
        return [f"falta {ruta_meta}"]
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    faltan = META_CLAVES - set(meta)
    if faltan:
        errores.append(f"modelo_meta.json: faltan claves {sorted(faltan)}")
    if not {"auc", "brier"} <= set(meta.get("metricas_oof", {})):
        errores.append("modelo_meta.json: metricas_oof debe tener auc y brier")
    if not (0 < meta.get("umbral_medio", 0) < meta.get("umbral_alto", 0) <= 1):
        errores.append("modelo_meta.json: se requiere 0 < umbral_medio < umbral_alto <= 1")
    if meta.get("n_niveles") != 3:
        errores.append("modelo_meta.json: n_niveles debe ser 3")
    for k in ("version", "semana_mas_reciente", "entrenado_hasta"):
        if not RE_FECHA.match(str(meta.get(k, ""))):
            errores.append(f"modelo_meta.json: '{k}' no es YYYY-MM-DD")

    dfs = {}
    for nombre in COLUMNAS:
        ruta = carpeta / nombre
        if not ruta.exists():
            errores.append(f"falta {nombre}")
            continue
        df = _leer(ruta, errores)
        if df is not None:
            dfs[nombre] = df

    if "scoring_modelo.csv" in dfs:
        sm = dfs["scoring_modelo.csv"]
        _check_serie(sm, "scoring_modelo.csv", meta, errores)
        por_ugt = sm.groupby("ugt")["semana"].nunique()
        if set(por_ugt.index) != UGT_IDS or (por_ugt < 2).any():
            errores.append("scoring_modelo.csv: se requieren las 4 UGTs con al menos 2 semanas cada una")
        if sm["semana"].max() != meta.get("semana_mas_reciente"):
            errores.append("scoring_modelo.csv: la semana máxima no coincide con meta.semana_mas_reciente")

    if "scoring_separadas.csv" in dfs:
        ss = dfs["scoring_separadas.csv"]
        _check_serie(ss, "scoring_separadas.csv", meta, errores)
        if not ss["origen"].isin(ORIGENES).all():
            errores.append(f"scoring_separadas.csv: origen inválido {sorted(set(ss['origen']) - ORIGENES)}")
        orden = ss.sort_values(["semana", "ugt"]).index
        if list(orden) != list(ss.index):
            errores.append("scoring_separadas.csv: no está ordenado por semana, ugt")
        for u, g in ss.groupby("ugt"):
            s = pd.to_datetime(g["semana"]).sort_values()
            if (s.diff().dropna() != pd.Timedelta(days=7)).any():
                errores.append(f"scoring_separadas.csv: la serie de {u} no es semanal continua")
        # OOF siempre antes que scoring en vivo
        if (ss["origen"] == "oof_backfill").any() and (ss["origen"] == "scoring_modelo").any():
            if ss.loc[ss["origen"] == "oof_backfill", "semana"].max() >= ss.loc[ss["origen"] == "scoring_modelo", "semana"].min():
                errores.append("scoring_separadas.csv: semanas oof_backfill posteriores a scoring_modelo")
        if "scoring_modelo.csv" in dfs:
            a = dfs["scoring_modelo.csv"].sort_values(["semana", "ugt"]).reset_index(drop=True)
            b = ss[ss["origen"] == "scoring_modelo"][COLUMNAS["scoring_modelo.csv"]].sort_values(["semana", "ugt"]).reset_index(drop=True)
            if not a.equals(b):
                errores.append("scoring_modelo.csv no coincide con las filas scoring_modelo de scoring_separadas.csv")

    if "features_historicas.csv" in dfs:
        fh = dfs["features_historicas.csv"]
        if not fh["ugt"].isin(UGT_IDS).all():
            errores.append("features_historicas.csv: ugt fuera de los 4 IDs")
        if not _fechas_validas(fh["semana"]):
            errores.append("features_historicas.csv: 'semana' con formato inválido")
        if fh.duplicated(["ugt", "semana"]).any():
            errores.append("features_historicas.csv: duplicados ugt+semana")
        for c in COLUMNAS["features_historicas.csv"][2:]:
            v = fh[c][fh[c] != ""]
            if not v.str.match(r"^\d+$").all():
                errores.append(f"features_historicas.csv: '{c}' debe ser entero >= 0 (o vacío)")
        if "scoring_modelo.csv" in dfs and fh["semana"].max() != dfs["scoring_modelo.csv"]["semana"].max():
            errores.append("features_historicas.csv: no llega a la semana más reciente del scoring")

    if "calibracion_modelo.csv" in dfs:
        cm = dfs["calibracion_modelo.csv"]
        if list(cm["nivel"]) != ["ALTO", "MEDIO", "BAJO"]:
            errores.append("calibracion_modelo.csv: debe tener exactamente las filas ALTO, MEDIO, BAJO")
        else:
            u = cm.set_index("nivel")["umbral"].astype(float)
            if abs(u["ALTO"] - meta["umbral_alto"]) > 1e-9 or abs(u["MEDIO"] - meta["umbral_medio"]) > 1e-9:
                errores.append("calibracion_modelo.csv: umbrales distintos a modelo_meta.json")
        for c in ("tasa_acierto", "recall"):
            v = pd.to_numeric(cm[c], errors="coerce")
            if v.isna().any() or ((v < 0) | (v > 1)).any():
                errores.append(f"calibracion_modelo.csv: '{c}' fuera de [0,1]")

    if "calendario_criticidad.csv" in dfs:
        cc = dfs["calendario_criticidad.csv"]
        el = cc["tipo"].str.lower() == "electoral"
        if not _fechas_validas(cc.loc[el, "fecha"]) if el.any() else False:
            errores.append("calendario_criticidad.csv: eventos electorales requieren 'fecha' YYYY-MM-DD")
        if not cc.loc[~el, "inicio_mes_dia"].str.match(RE_MMDD).all():
            errores.append("calendario_criticidad.csv: eventos recurrentes requieren 'inicio_mes_dia' MM-DD")
        if not cc["usar_como_proxima_fecha_critica"].isin({"true", "false"}).all():
            errores.append("calendario_criticidad.csv: usar_como_proxima_fecha_critica debe ser true/false")
        if not cc["criticidad"].isin({"ALTO", "MEDIO-ALTO", "MEDIO"}).all():
            errores.append(f"calendario_criticidad.csv: criticidad fuera de ALTO/MEDIO-ALTO/MEDIO: "
                           f"{sorted(set(cc['criticidad']) - {'ALTO', 'MEDIO-ALTO', 'MEDIO'})}")
        for v in cc["ugts"]:
            if v != "TODAS" and not set(v.split(",")) <= UGT_IDS:
                errores.append(f"calendario_criticidad.csv: ugts inválido {v!r}")

    if "alertas_electorales_2026.csv" in dfs:
        ae = dfs["alertas_electorales_2026.csv"]
        if (ae == "").all(axis=1).any():
            errores.append("alertas_electorales_2026.csv: filas vacías")

    if "eventos_historicos.csv" in dfs:
        eh = dfs["eventos_historicos.csv"]
        if not eh["ugt"].isin(UGT_IDS).all():
            errores.append("eventos_historicos.csv: ugt fuera de los 4 IDs")
        if not _fechas_validas(eh["fecha"]):
            errores.append("eventos_historicos.csv: 'fecha' inválida")
        if not eh["fuente_url"].map(lambda u: u == "" or u.startswith("http")).all():
            errores.append("eventos_historicos.csv: fuente_url debe ser vacía o http(s)")
        if not eh["severidad"].isin({"", "ALTO", "MEDIO", "BAJO"}).all():
            errores.append("eventos_historicos.csv: severidad fuera de ALTO/MEDIO/BAJO")
        if (eh["evento"] == "").any():
            errores.append("eventos_historicos.csv: eventos sin descripción")

    if "actores_criticos.csv" in dfs:
        ac = dfs["actores_criticos.csv"]
        if not _fechas_validas(ac["ultima_mencion"]):
            errores.append("actores_criticos.csv: 'ultima_mencion' inválida")
        if not ac["n_apariciones"].str.match(r"^\d+$").all():
            errores.append("actores_criticos.csv: n_apariciones debe ser entero")
        if ac["nombre"].duplicated().any():
            errores.append("actores_criticos.csv: nombres duplicados")
    return errores


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "export_dashboard"
    errores = validar(carpeta)
    if errores:
        print(f"✗ {len(errores)} problema(s) en {carpeta}:")
        for e in errores:
            print("  -", e)
        raise SystemExit(1)
    print(f"✓ {carpeta} cumple el contrato del dashboard")


if __name__ == "__main__":
    main()
