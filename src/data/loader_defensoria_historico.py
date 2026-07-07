"""Loader histórico de la Defensoría del Pueblo — conflictos sociales de Áncash
en las 4 UGTs de ANTAMINA, extraídos de los Reportes Mensuales de Conflictos
Sociales (PDF).

Motivación (ver memoria del proyecto): la fuente de prensa
`dataHistoricaProtestas.xlsx` subregistra el periodo 2016-2020 (aparece en
cero pese a que Áncash fue la región más conflictiva del país esos años). La
Defensoría publica mensualmente el estado de CADA conflicto activo/latente por
distrito, con cobertura completa — resuelve ese hueco de cronicidad.

Distinción clave (CLAUDE.md):
  - Estado "activo/latente" de un conflicto crónico  → feature de CONTEXTO
    (cronicidad por UGT/mes). NO es el label (usar la mera existencia del
    conflicto como target lo volvería trivialmente positivo).
  - Acción colectiva de protesta con fecha (paro, bloqueo, movilización)
    → candidato a LABEL (evento nuevo). La Defensoría los reporta de forma
    rala, así que esta salida complementa, no reemplaza, la BD de incidentes.

Salidas:
  data/raw/defensoria/*.pdf                        (PDFs descargados)
  data/interim/defensoria_hist_conflictos.parquet  (1 fila = reporte×conflicto UGT)

Uso:
  python -m src.data.loader_defensoria_historico --desde 2021-01 --hasta 2022-12
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/defensoria")
INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "defensoria_hist_conflictos.parquet"

BASE = "https://www.defensoria.gob.pe/wp-content/uploads"

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# ── distrito → UGT (réplica de ancash_datos.JERARQUIA, normalizado) ───────────
DIST2UGT = {
    "san marcos": "Mina San Marcos", "chavin de huantar": "Mina San Marcos",
    "huachis": "Mina San Marcos", "san pedro de chana": "Mina San Marcos",
    "huallanca": "Huallanca", "aquia": "Huallanca", "chiquian": "Huallanca",
    "cajacay": "Valle Fortaleza", "antonio raimondi": "Valle Fortaleza",
    "colquioc": "Valle Fortaleza", "huayllacayan": "Valle Fortaleza",
    "catac": "Valle Fortaleza", "pampas chico": "Valle Fortaleza",
    "marca": "Valle Fortaleza", "llacllin": "Valle Fortaleza",
    "pararin": "Valle Fortaleza", "paramonga": "Valle Fortaleza",
    "huarmey": "Huarmey",
}
DIST_RX = {d: re.compile(r"\b" + re.escape(d) + r"\b") for d in DIST2UGT}

PROTESTA_RX = re.compile(r"\b(paro|bloque|moviliza|marcha|planton|plant[oó]n|"
                         r"huelga|toma de|protesta|manifestaci)", re.I)
NO_HECHOS_RX = re.compile(r"no se registraron nuevos hechos", re.I)
# fechas tipo "El 31 de agosto", "el 5 de enero de 2022"
FECHA_RX = re.compile(r"\bel\s+(\d{1,2})\s+de\s+(" + "|".join(MESES) + r")(?:\s+de\s+(\d{4}))?", re.I)

# Nº de reporte por (año, mes-dato). Ancla: N°143 = enero 2016; +1 por mes.
def _num_reporte(anio: int, mes: int) -> int:
    return 143 + (anio - 2016) * 12 + (mes - 1)


def _norm(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


# ── Descarga ─────────────────────────────────────────────────────────────────
def _candidatos_url(anio: int, mes: int) -> list[str]:
    """Genera variantes de URL probables (el nombre de archivo es inconsistente
    entre reportes). Se prueban en orden hasta que una responda 200."""
    num = _num_reporte(anio, mes)
    mesn = MESES[mes - 1]
    pub_anio, pub_mes = (anio, mes + 1) if mes < 12 else (anio + 1, 1)
    folder = f"{pub_anio}/{pub_mes:02d}"
    grado = "%C2%B0"   # °
    variantes = [
        # formato WordPress (2019+): /wp-content/uploads/{pub_año}/{pub_mes}/
        f"{BASE}/{folder}/Reporte-Mensual-de-Conflictos-Sociales-N{grado}-{num}-{mesn}-{anio}.pdf",
        f"{BASE}/{folder}/Reporte-Mensual-de-Conflictos-Sociales-N{grado}-{num}-{mesn.capitalize()}-{anio}.pdf",
        f"{BASE}/{folder}/Reporte-Mensual-de-Conflictos-Sociales-n.{grado}-{num}-%E2%80%93-{mesn}-{anio}.pdf",
        f"{BASE}/{folder}/Reporte-Mensual-de-Conflictos-Sociales-n.{grado}-{num}-{mesn}-{anio}.pdf",
        f"{BASE}/{folder}/Reporte-Mensual-de-Conflictos-Sociales-N-{num}-{mesn.capitalize()}-{anio}.pdf",
        # formato 2019: "Conflictos-Sociales-N°-{num}-{Mes}-2019.pdf"
        f"{BASE}/{folder}/Conflictos-Sociales-N{grado}-{num}-{mesn.capitalize()}-{anio}.pdf",
        f"{BASE}/{folder}/Conflictos-Sociales-N-{num}-{mesn.capitalize()}-{anio}.pdf",
    ]
    # formato antiguo (2016-2018): /modules/Downloads/conflictos/{año}/  con "N-{num}---{Mes}"
    old = "https://www.defensoria.gob.pe/modules/Downloads/conflictos"
    variantes += [
        f"{old}/{anio}/Reporte-Mensual-de-Conflictos-Sociales-N-{num}---{mesn.capitalize()}-{anio}.pdf",
        f"{old}/{anio}/Reporte-Mensual-de-Conflictos-Sociales-N-{num}-{mesn.capitalize()}-{anio}.pdf",
        f"{old}/{anio}/Reporte-Mensual-de-Conflictos-Sociales-N{grado}-{num}-{mesn.capitalize()}-{anio}.pdf",
    ]
    if mesn == "septiembre":  # variante "setiembre"
        variantes += [v.replace("septiembre", "setiembre") for v in variantes]
    return variantes


# URLs confirmadas manualmente (evitan el tanteo de variantes)
URLS_CONFIRMADAS = {
    (2021, 1): f"{BASE}/2021/02/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-203-enero-2021.pdf",
    (2021, 2): f"{BASE}/2021/03/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-204-febrero-2021.pdf",
    (2021, 3): f"{BASE}/2021/04/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-205-marzo-2021.pdf",
    (2021, 4): f"{BASE}/2021/05/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-206-abril-2021.pdf",
    (2021, 6): f"{BASE}/2021/07/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-208-junio-2021.pdf",
    (2021, 7): f"{BASE}/2021/08/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-209-julio-2021.pdf",
    (2021, 8): f"{BASE}/2021/09/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-210-agosto-2021.pdf",
    (2018, 1): f"{BASE}/2018/02/Reporte-Mensual-de-Conflictos-Sociales-N-167-Enero-2018.pdf",
    (2018, 2): f"{BASE}/2018/07/Reporte-Mensual-de-Conflictos-Sociales-N-168-Febrero-2018.pdf",
    (2018, 4): f"{BASE}/2018/07/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-170-Abril-2018.pdf",
    (2018, 6): f"{BASE}/2018/07/Reporte-Mensual-de-Conflictos-Sociales-N-172-Junio-2018.pdf",
    (2018, 7): f"{BASE}/2018/08/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-173-Julio-2018.pdf",
    (2021, 10): f"{BASE}/2021/11/Reporte-Mensual-de-Conflictos-Sociales-n.%C2%B0-212-octubre-2021.pdf",
    (2021, 11): f"{BASE}/2021/12/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-213-noviembre-2021.pdf",
    (2021, 12): f"{BASE}/2022/01/Reporte-Mensual-de-Conflictos-Sociales-n.%C2%B0-214-%E2%80%93-diciembre-2021.pdf",
    (2022, 1): f"{BASE}/2022/02/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-215-Enero-2022-1.pdf",
    (2022, 8): f"{BASE}/2022/09/Reporte-Mensual-de-Conflictos-Sociales-N%C2%B0-222-Agosto-2022.pdf",
}


def _descargar(anio: int, mes: int) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destino = RAW_DIR / f"reporte_{anio}_{mes:02d}.pdf"
    if destino.exists() and destino.stat().st_size > 10_000:
        return destino
    urls = []
    if (anio, mes) in URLS_CONFIRMADAS:
        urls.append(URLS_CONFIRMADAS[(anio, mes)])
    urls += _candidatos_url(anio, mes)
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) > 10_000 and data[:4] == b"%PDF":
                destino.write_bytes(data)
                print(f"  ✓ {anio}-{mes:02d}  ({len(data)//1024} KB)  {url.split('/')[-1]}")
                return destino
        except Exception:
            continue
    print(f"  ✗ {anio}-{mes:02d}  NO se pudo descargar (probé {len(urls)} variantes)")
    return None


# ── Parseo ───────────────────────────────────────────────────────────────────
def parsear_reporte(pdf: Path, anio: int, mes: int) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
    full = "\n".join((p.extract_text() or "") for p in reader.pages)
    rows = []
    for b in re.split(r"(?=Tipo:\s)", full):
        bn = _norm(b)
        if "ancash" not in bn:
            continue
        dists = [d for d, rx in DIST_RX.items() if rx.search(bn)]
        if not dists:
            continue
        for ugt in sorted(set(DIST2UGT[d] for d in dists)):
            ugt_dists = [d for d in dists if DIST2UGT[d] == ugt]
            tipo = (re.search(r"Tipo:\s*([^\n]+)", b) or [None, ""])[1].strip().rstrip(".")
            ingreso = (re.search(r"Ingres[oó] como caso nuevo:\s*([^\n]+)", b) or [None, ""])[1].strip()
            es_antamina = "antamina" in bn
            sin_hechos = bool(NO_HECHOS_RX.search(b))
            posible_protesta = bool(PROTESTA_RX.search(b)) and not sin_hechos
            rows.append({
                "anio": anio, "mes": mes,
                "num_reporte": _num_reporte(anio, mes),
                "ugt": ugt, "distritos": ",".join(ugt_dists),
                "tipo": tipo, "ingreso": ingreso,
                "es_antamina": es_antamina,
                "conflicto_activo": True,            # aparece en el reporte
                "sin_hechos_mes": sin_hechos,
                "posible_protesta": posible_protesta,
            })
    return rows


# ── Orquestación ─────────────────────────────────────────────────────────────
def _rango(desde: str, hasta: str):
    d = pd.Period(desde, "M"); h = pd.Period(hasta, "M")
    p = d
    while p <= h:
        yield p.year, p.month
        p += 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2021-01")
    ap.add_argument("--hasta", default="2022-12")
    args = ap.parse_args()

    print(f"Descargando reportes Defensoría {args.desde} → {args.hasta}")
    faltantes = []
    for anio, mes in _rango(args.desde, args.hasta):
        if _descargar(anio, mes) is None:
            faltantes.append(f"{anio}-{mes:02d}")

    # Parsear TODOS los PDFs en RAW_DIR (acumula entre corridas de distinto rango)
    todas = []
    for pdf in sorted(RAW_DIR.glob("reporte_*.pdf")):
        m = re.match(r"reporte_(\d{4})_(\d{2})\.pdf", pdf.name)
        if not m:
            continue
        a, me = int(m.group(1)), int(m.group(2))
        try:
            todas.extend(parsear_reporte(pdf, a, me))
        except Exception as e:
            print(f"  ! error parseando {a}-{me:02d}: {e}")

    df = pd.DataFrame(todas)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)
    print(f"\n✓ {len(df)} filas (reporte×conflicto-UGT) → {SALIDA}")
    if faltantes:
        print(f"  Reportes no obtenidos: {faltantes}")
    if len(df):
        print("\nConflictos activos por UGT (nº de reporte-mes en que aparece):")
        print(df.groupby("ugt").size().to_string())
        print(f"\nMeses con posible protesta (hecho nuevo): {int(df['posible_protesta'].sum())}")
        print(f"Conflictos Antamina-dirigidos (filas): {int(df['es_antamina'].sum())}")


if __name__ == "__main__":
    main()
