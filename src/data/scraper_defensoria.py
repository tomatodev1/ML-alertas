import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Constantes ────────────────────────────────────────────────────────────────
URL = (
    "https://wabi-us-east2-b-primary-api.analysis.windows.net"
    "/public/reports/querydata?synchronous=true"
)

HEADERS = {
    "Content-Type": "application/json;charset=UTF-8",
    "X-PowerBI-ResourceKey": "95611526-9781-4044-9590-2b21e0fa8c79",
}

ZONAS_CON_TILDE = [
    "Áncash", "Huánuco", "Pasco", "Cajamarca", "La Libertad", "Ica", "Lima Provincias",
]
ZONAS_SIN_TILDE = [
    "Ancash", "Huanuco", "Pasco", "Cajamarca", "La Libertad", "Ica", "Lima Provincias",
]

MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

AÑOS = [2024, 2025, 2026]

SLEEP = 1.5  # segundos entre POSTs

SALIDA = Path("data/raw/defensoria/defensoria_historico.csv")

# Fecha de corte: incluir hasta este mes/año (inclusive)
AÑO_CORTE, MES_CORTE = 2026, 6


# ── Constructores de payload ───────────────────────────────────────────────────
def _filtro_estado() -> list[dict]:
    return [{"Condition": {"In": {
        "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "b"}}, "Property": "Estado"}}],
        "Values": [
            [{"Literal": {"Value": "'Activo'"}}],
            [{"Literal": {"Value": "'Nuevo'"}}],
            [{"Literal": {"Value": "'Latente'"}}],
        ],
    }}}]


def payload_global(año: int, mes: str) -> dict:
    """KPIs globales del mes: total en escalamiento y violencia (sin filtro de zona)."""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {"Commands": [{
                "SemanticQueryDataShapeCommand": {
                    "Query": {
                        "Version": 2,
                        "From": [
                            {"Name": "b", "Entity": "BASE TOTAL", "Type": 0},
                            {"Name": "l", "Entity": "LocalDateTable_94ae2701-59c4-47e0-95f8-ef1669449fc6", "Type": 0},
                            {"Name": "c", "Entity": "CALENDARIO", "Type": 0},
                        ],
                        "Select": [{
                            "Measure": {
                                "Expression": {"SourceRef": {"Source": "b"}},
                                "Property": "HTML_Escalamiento_Fit",
                            },
                            "Name": "BASE TOTAL.HTML_Escalamiento_Small",
                            "NativeReferenceName": "HTML_Escalamiento_Fit",
                        }],
                        "Where": [
                            {"Condition": {"In": {
                                "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "l"}}, "Property": "Año"}}],
                                "Values": [[{"Literal": {"Value": f"{año}L"}}]],
                            }}},
                            {"Condition": {"In": {
                                "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "c"}}, "Property": "NombreMes"}}],
                                "Values": [[{"Literal": {"Value": f"'{mes}'"}}]],
                            }}},
                            *_filtro_estado(),
                        ],
                    },
                    "Binding": {
                        "Primary": {"Groupings": [{"Projections": [0]}]},
                        "DataReduction": {"DataVolume": 3, "Primary": {"Top": {}}},
                        "Version": 1,
                    },
                    "ExecutionMetricsKind": 1,
                },
            }]},
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": "15990f8c-699f-4a81-9813-1b0aca3f624c",
                "Sources": [{"ReportId": "4b4134a5-81e8-418c-ba86-10f43e5a6b8b", "VisualId": "30c3c5b80636c60755e6"}],
            },
        }],
        "cancelQueries": [],
        "modelId": 7024015,
    }


def payload_escalamiento_zona(año: int, mes: str, zona: str) -> dict:
    """Escalamiento filtrado por departamento (HTML_Escalamiento_Fit + filtro Dpto. 1).
    El HTML responde: 'En Escalamiento ⚠️ {actual} {tendencia} {delta%} {mes_ant} {valor_ant}'
    → nums[0] = escalamiento actual, nums[-1] = escalamiento mes anterior.
    """
    p = payload_global(año, mes)
    filtro_dpto = {"Condition": {"In": {
        "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "b"}}, "Property": "Dpto. 1"}}],
        "Values": [[{"Literal": {"Value": f"'{zona}'"}}]],
    }}}
    p["queries"][0]["Query"]["Commands"][0]["SemanticQueryDataShapeCommand"]["Query"]["Where"].insert(0, filtro_dpto)
    return p


# ── HTTP ──────────────────────────────────────────────────────────────────────
def post_query(payload: dict) -> dict | None:
    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ERROR POST: {e}")
        return None


# ── Parsing HTML ──────────────────────────────────────────────────────────────
def _extraer_m0(resp: dict) -> str | None:
    try:
        return resp["results"][0]["result"]["data"]["dsr"]["DS"][0]["PH"][0]["DM0"][0]["M0"]
    except (KeyError, IndexError, TypeError):
        return None


def _numeros(html: str) -> list[int]:
    texto = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return [int(n) for n in re.findall(r"\b\d+\b", texto)]


def extraer_escalamiento(html: str) -> int | None:
    nums = _numeros(html)
    return nums[0] if nums else None


def extraer_escalamiento_zona(html: str) -> tuple[int | None, int | None]:
    """Devuelve (escalamiento_actual, escalamiento_mes_anterior) del KPI card por zona."""
    nums = _numeros(html)
    if not nums:
        return None, None
    actual = nums[0]
    # El último número es el valor del mes anterior; si hay solo uno, prev=actual
    prev = nums[-1] if len(nums) > 1 else actual
    return actual, prev


# ── Detección de formato de zona ──────────────────────────────────────────────
def detectar_zonas() -> list[str]:
    """Detecta si la API acepta nombres con o sin tilde usando diciembre 2025."""
    print("Detectando formato de nombres de zona...")
    for zonas in (ZONAS_CON_TILDE, ZONAS_SIN_TILDE):
        resp = post_query(payload_escalamiento_zona(2025, "diciembre", zonas[0]))
        time.sleep(SLEEP)
        if resp:
            html = _extraer_m0(resp)
            if html:
                # Con nombre correcto devuelve KPI card con número; ambas formas
                # pueden devolver 0, así que comparamos con la suma global del mes.
                resp_g = post_query(payload_global(2025, "diciembre"))
                time.sleep(SLEEP)
                html_g = _extraer_m0(resp_g) if resp_g else None
                esc_g = extraer_escalamiento(html_g) if html_g else None
                if esc_g is not None:
                    etiqueta = "con tilde" if zonas is ZONAS_CON_TILDE else "sin tilde"
                    print(f"  Global dic-2025 escalamiento={esc_g}. Usando nombres {etiqueta}.")
                    return list(zonas)
    print("  AVISO: no se pudo detectar formato. Usando con tilde por defecto.")
    return list(ZONAS_CON_TILDE)


# ── Principal ─────────────────────────────────────────────────────────────────
def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    ZONAS = detectar_zonas()

    registros: list[dict] = []

    for año in AÑOS:
        for mes_idx, mes in enumerate(MESES, start=1):
            if año > AÑO_CORTE or (año == AÑO_CORTE and mes_idx > MES_CORTE):
                continue

            # KPI global del mes
            resp_global = post_query(payload_global(año, mes))
            time.sleep(SLEEP)
            esc_global = None
            if resp_global:
                html_g = _extraer_m0(resp_global)
                if html_g:
                    esc_global = extraer_escalamiento(html_g)

            # Escalamiento por zona
            for zona in ZONAS:
                resp_z = post_query(payload_escalamiento_zona(año, mes, zona))
                time.sleep(SLEEP)
                esc_zona = esc_zona_prev = None
                if resp_z:
                    html_z = _extraer_m0(resp_z)
                    if html_z:
                        esc_zona, esc_zona_prev = extraer_escalamiento_zona(html_z)

                print(f"[{año}-{mes}] {zona} → esc_zona={esc_zona}, esc_global={esc_global}")

                registros.append({
                    "año": año,
                    "mes": mes,
                    "mes_num": mes_idx,
                    "zona": zona,
                    "escalamiento_zona": esc_zona,
                    "escalamiento_zona_prev": esc_zona_prev,
                    "escalamiento_global": esc_global,
                })

    df = pd.DataFrame(registros)
    df.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"\n✓ Extracción completa: {len(df)} filas guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
