"""Loader de reportes situacionales semanales/mensuales de Antamina (Word/PowerPoint).

Extrae texto de src/data/2025/ y src/data/2026/ (reportes "VP Antamina"),
busca menciones de los 18 distritos de interés (vía la jerarquía de
ancash_datos.py) y etiqueta cada párrafo con: si menciona a Antamina
explícitamente, y si tiene tono de tensión (cuestionan/denuncian/etc.) o de
compromiso/mesa de diálogo.

Es texto narrativo escrito por el equipo de relaciones de Antamina — la
única fuente del proyecto con "compromiso de las partes" explícito (mesas de
diálogo, acuerdos, cronogramas), marcado como "PENDIENTE DE FUENTE" en el
dashboard desde el inicio del proyecto.
"""
import re
import sys
from pathlib import Path

import docx
import pandas as pd
import pptx

from src.scoring.ancash_datos import JERARQUIA, _norm

CARPETAS = [Path("src/data") / "2025", Path("src/data") / "2026"]
INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "reportes_antamina_parrafos.parquet"

PAT_FECHA = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})")
PALABRAS_NEGATIVO = ["cuestion", "denuncia", "protest", "exig", "critic", "alert",
                      "reclam", "rechaz", "bloque", "paraliz", "amenaz"]
PALABRAS_COMPROMISO = ["mesa de dialogo", "compromiso", "acuerdo", "coordina",
                        "avance", "socializ", "cronograma"]


def _extraer_docx(ruta: Path) -> str:
    doc = docx.Document(ruta)
    lineas = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for t in doc.tables:
        for row in t.rows:
            celda = " | ".join(c.text.strip() for c in row.cells)
            if celda.strip(" |"):
                lineas.append(celda)
    return "\n".join(lineas)


def _extraer_pptx(ruta: Path) -> str:
    prs = pptx.Presentation(ruta)
    lineas = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                lineas.append(shape.text_frame.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    celda = " | ".join(c.text.strip() for c in row.cells)
                    if celda.strip(" |"):
                        lineas.append(celda)
    return "\n".join(lineas)


def _fecha_de_nombre(nombre: str) -> pd.Timestamp | None:
    """Fecha del reporte = ÚLTIMA fecha del nombre. Los reportes mensuales
    nuevos se nombran por periodo ("09.06.26 al 09.07.26"); usar la fecha
    inicial fecharía el reporte antes de que existiera (fuga temporal)."""
    matches = PAT_FECHA.findall(nombre)
    if not matches:
        return None
    dd, mm, yy = matches[-1]
    try:
        return pd.Timestamp(year=2000 + int(yy), month=int(mm), day=int(dd))
    except ValueError:
        return None


def _vigencia_dias(nombre: str) -> int:
    """Días que el contenido del reporte se considera vigente en las features
    `rep_*_4w`: 28 (ventana de 4 semanas) para reportes de una sola fecha; el
    largo del periodo (mínimo 28) para mensuales "dd.mm.aa al dd.mm.aa"."""
    matches = PAT_FECHA.findall(nombre)
    if len(matches) >= 2:
        try:
            f = [pd.Timestamp(year=2000 + int(y), month=int(m), day=int(d)) for d, m, y in (matches[0], matches[-1])]
            return max(28, int((f[1] - f[0]).days))
        except ValueError:
            pass
    return 28


def parsear_reportes() -> pd.DataFrame:
    dist2ugt = {_norm(d): ugt for ugt, _, d in JERARQUIA}
    distritos_ordenados = sorted(dist2ugt.keys(), key=len, reverse=True)

    filas = []
    for carpeta in CARPETAS:
        if not carpeta.exists():
            continue
        for archivo in sorted(carpeta.rglob("*")):
            if archivo.suffix.lower() not in (".docx", ".pptx"):
                continue
            fecha = _fecha_de_nombre(archivo.stem)
            if fecha is None:
                continue
            vigencia = _vigencia_dias(archivo.stem)
            try:
                texto = _extraer_docx(archivo) if archivo.suffix.lower() == ".docx" else _extraer_pptx(archivo)
            except Exception:
                continue

            for parrafo in texto.split("\n"):
                if len(parrafo.strip()) <= 25:
                    continue
                pn = _norm(parrafo)
                ugts_match = {dist2ugt[dn] for dn in distritos_ordenados
                              if re.search(r"\b" + re.escape(dn) + r"\b", pn)}
                if not ugts_match:
                    continue
                menciona_antamina = "antamina" in pn
                tono_negativo = any(k in pn for k in PALABRAS_NEGATIVO)
                tono_compromiso = any(k in pn for k in PALABRAS_COMPROMISO)
                for ugt in ugts_match:
                    filas.append({
                        "fecha_reporte": fecha, "ugt": ugt,
                        "menciona_antamina": menciona_antamina,
                        "tono_negativo": tono_negativo,
                        "tono_compromiso": tono_compromiso,
                        "vigencia_dias": vigencia,
                        "archivo": archivo.name,
                        "texto": parrafo.strip()[:600],
                    })
    return pd.DataFrame(filas)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = parsear_reportes()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)
    print(f"✓ Reportes Antamina procesados: {len(df)} párrafos con distrito identificado")
    if not df.empty:
        print(f"  Rango: {df['fecha_reporte'].min().date()} → {df['fecha_reporte'].max().date()}")
        print(df.groupby("ugt")[["menciona_antamina", "tono_negativo", "tono_compromiso"]].sum())
    print(f"  Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
