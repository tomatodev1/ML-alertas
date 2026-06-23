import sys
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "calendario.parquet"

# ── Rango de generación ───────────────────────────────────────────────────────
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2026-12-31"

# ── Eventos electorales ───────────────────────────────────────────────────────
ELECCIONES = [
    "2024-10-06",  # ERM 2024 - Elecciones Regionales y Municipales
    "2026-04-11",  # Elecciones Generales 2026 - Primera vuelta
    "2026-06-07",  # Elecciones Generales 2026 - Segunda vuelta
    "2026-10-04",  # ERM 2026 - Elecciones Regionales y Municipales
]

# ── Fechas críticas ───────────────────────────────────────────────────────────
# Las ERM generan conflictos locales semanas antes (candidatos movilizando
# comunidades, disputas por obras, tensiones en zonas como Áncash y Cajamarca).
FECHAS_CRITICAS = [
    # 4 semanas pre-ERM 2024
    "2024-09-08",
    "2024-09-15",
    "2024-09-22",
    "2024-09-29",
    # 4 semanas pre-ERM 2026
    "2026-09-06",
    "2026-09-13",
    "2026-09-20",
    "2026-09-27",
]

# ── Feriados nacionales del Perú 2024-2026 ────────────────────────────────────
# Incluye días de elección (declarados feriados por el JNE/ONPE).
# Semana Santa: Easter 2024 = 31 mar, 2025 = 20 abr, 2026 = 5 abr.
FERIADOS = {
    # 2024
    "2024-01-01",  # Año Nuevo
    "2024-03-28",  # Jueves Santo
    "2024-03-29",  # Viernes Santo
    "2024-05-01",  # Día del Trabajo
    "2024-06-29",  # San Pedro y San Pablo
    "2024-07-28",  # Fiestas Patrias
    "2024-07-29",  # Gran Parada Militar
    "2024-08-30",  # Santa Rosa de Lima
    "2024-10-06",  # Día de elección ERM 2024
    "2024-10-08",  # Combate de Angamos
    "2024-11-01",  # Día de Todos los Santos
    "2024-12-08",  # Inmaculada Concepción
    "2024-12-25",  # Navidad
    # 2025
    "2025-01-01",
    "2025-04-17",  # Jueves Santo
    "2025-04-18",  # Viernes Santo
    "2025-05-01",
    "2025-06-29",
    "2025-07-28",
    "2025-07-29",
    "2025-08-30",
    "2025-10-08",
    "2025-11-01",
    "2025-12-08",
    "2025-12-25",
    # 2026
    "2026-01-01",
    "2026-04-02",  # Jueves Santo
    "2026-04-03",  # Viernes Santo
    "2026-04-11",  # Primera vuelta generales
    "2026-05-01",
    "2026-06-07",  # Segunda vuelta generales (también Día de la Bandera)
    "2026-06-29",
    "2026-07-28",
    "2026-07-29",
    "2026-08-30",
    "2026-10-04",  # Día de elección ERM 2026
    "2026-10-08",
    "2026-11-01",
    "2026-12-08",
    "2026-12-25",
}

MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


# ── Generación ────────────────────────────────────────────────────────────────
def generar_calendario() -> pd.DataFrame:
    elecciones_ts = pd.to_datetime(ELECCIONES)
    feriados_ts = pd.to_datetime(sorted(FERIADOS))
    criticas_ts = pd.to_datetime(FECHAS_CRITICAS)

    # Una fila por semana; semana_inicio = lunes
    semanas = pd.date_range(start=FECHA_INICIO, end=FECHA_FIN, freq="W-MON")

    filas: list[dict] = []
    for lunes in semanas:
        domingo = lunes + pd.Timedelta(days=6)

        # Feriados que caen dentro de la semana [lunes, domingo]
        n_feriados = int(((feriados_ts >= lunes) & (feriados_ts <= domingo)).sum())

        # ±1 semana alrededor de cualquier elección
        # → la elección cae en [lunes − 7d, domingo + 7d]
        es_electoral = int(any(
            lunes - pd.Timedelta(days=7) <= e <= domingo + pd.Timedelta(days=7)
            for e in elecciones_ts
        ))

        # Alguna fecha crítica cae en la semana
        es_critica = int(((criticas_ts >= lunes) & (criticas_ts <= domingo)).any())

        # Días desde el lunes hasta la próxima elección (-1 si ya pasaron todas)
        futuras = elecciones_ts[elecciones_ts >= lunes]
        dias_hasta = int((futuras.min() - lunes).days) if len(futuras) > 0 else -1

        filas.append({
            "semana_inicio": lunes,
            "año": lunes.year,
            "mes": lunes.month,
            "mes_nombre": MESES_ES[lunes.month],
            "trimestre": (lunes.month - 1) // 3 + 1,
            "semana_iso": lunes.isocalendar().week,
            "n_feriados": n_feriados,
            "es_semana_electoral": es_electoral,
            "es_fecha_critica": es_critica,
            "dias_hasta_eleccion": dias_hasta,
        })

    return pd.DataFrame(filas)


# ── Guardar ───────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)


# ── Resumen ───────────────────────────────────────────────────────────────────
def _resumen(df: pd.DataFrame) -> None:
    inicio = df["semana_inicio"].min().date()
    fin = df["semana_inicio"].max().date()
    n_electorales = df["es_semana_electoral"].sum()
    n_criticas = df["es_fecha_critica"].sum()
    n_feriados = len(FERIADOS)

    print(f"✓ Calendario generado: {len(df)} semanas ({inicio} → {fin})")
    print(f"  Semanas electorales (±1 semana): {n_electorales}")
    print(f"  Semanas con fecha crítica:        {n_criticas}")
    print(f"  Feriados nacionales registrados:  {n_feriados}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = generar_calendario()
    guardar(df)
    _resumen(df)


if __name__ == "__main__":
    main()
