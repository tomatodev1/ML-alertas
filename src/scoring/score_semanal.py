"""
Script que corre cada lunes y calcula el riesgo de conflicto para cada zona
de Track B en las próximas HORIZONTE_DIAS. Escribe los resultados a la tabla
`riesgo_zona_semana` en Neon (PostgreSQL).
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.dataset.build_master import (
    _cargar_defensoria,
    _cargar_incidentes,
    _features_alertas,
    _features_defensoria,
    _features_incidentes,
    _floor_to_monday,
    _join_inei,
)

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────
MODELO_PATH = Path("models/modelo_v1_track_B.pkl")
DATABASE_URL = os.getenv("DATABASE_URL")  # Neon connection string
HORIZONTE_DIAS = 30
SEMANAS_HISTORIA = 60  # buffer para rolling 4w y racha consecutiva sin truncar

INTERIM_DIR = Path("data/interim")

ZONAS_TRACK_B = ["Ica", "Pisco", "Huarmey", "Barranca", "Lima Provincias"]

# Router zona → cliente (igual que router.py del sistema actual)
ZONA_CLIENTE = {
    "Ica":             ["Agrícola Chapi", "Clientes Sur"],
    "Pisco":           ["Pisco"],
    "Huarmey":         ["Agrícola Huarmey"],
    "Barranca":        ["Agrícola Santa Azul", "Agrícola Huarmey"],
    "Lima Provincias": ["Agrícola Santa Azul"],
}

MODELO_VERSION = "v1_track_B_lgbm"


# ── Semana actual ─────────────────────────────────────────────────────────────
def _semana_actual() -> pd.Timestamp:
    return _floor_to_monday(pd.Series([pd.Timestamp.today()])).iloc[0]


def _construir_skeleton_reciente(lunes_actual: pd.Timestamp, zonas: list[str], n_semanas: int) -> pd.DataFrame:
    semanas = pd.date_range(end=lunes_actual, periods=n_semanas, freq="W-MON")
    filas = [(z, s) for z in zonas for s in semanas]
    return pd.DataFrame(filas, columns=["zona", "semana_inicio"])


# ── Features (reutiliza build_master.py, no duplica lógica) ─────────────────
def calcular_features_actuales(lunes_actual: pd.Timestamp) -> pd.DataFrame:
    alertas = pd.read_parquet(INTERIM_DIR / "alertas_normalizadas.parquet")
    alertas["fecha"] = pd.to_datetime(alertas["fecha"])
    hoy = pd.Timestamp.today().normalize()
    alertas = alertas[alertas["fecha"] <= hoy]  # anti-fuga defensivo

    calendario = pd.read_parquet(INTERIM_DIR / "calendario.parquet")
    calendario["semana_inicio"] = pd.to_datetime(calendario["semana_inicio"])

    inei = pd.read_parquet(INTERIM_DIR / "inei_pobreza.parquet")
    defensoria = _cargar_defensoria()
    incidentes = _cargar_incidentes()

    skeleton = _construir_skeleton_reciente(lunes_actual, ZONAS_TRACK_B, SEMANAS_HISTORIA)

    cal_cols = [c for c in calendario.columns if c != "semana_inicio"]
    master = skeleton.merge(calendario[["semana_inicio"] + cal_cols], on="semana_inicio", how="left")

    feat_alertas = _features_alertas(skeleton[["zona", "semana_inicio"]].copy(), alertas)
    alert_cols = [c for c in feat_alertas.columns if c not in ["zona", "semana_inicio"]]
    master = master.merge(
        feat_alertas[["zona", "semana_inicio"] + alert_cols], on=["zona", "semana_inicio"], how="left"
    )

    feat_incidentes = _features_incidentes(skeleton[["zona", "semana_inicio"]].copy(), incidentes)
    inc_cols = [c for c in feat_incidentes.columns if c not in ["zona", "semana_inicio"]]
    master = master.merge(
        feat_incidentes[["zona", "semana_inicio"] + inc_cols], on=["zona", "semana_inicio"], how="left"
    )

    master = _features_defensoria(master, defensoria)
    master = _join_inei(master, inei)

    actual = master[master["semana_inicio"] == lunes_actual].copy()
    if len(actual) != len(ZONAS_TRACK_B):
        faltantes = set(ZONAS_TRACK_B) - set(actual["zona"])
        raise RuntimeError(f"No se pudieron calcular features para todas las zonas. Faltan: {faltantes}")
    return actual


# ── Modelo ────────────────────────────────────────────────────────────────────
def cargar_modelo() -> dict:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODELO_PATH}")
    return joblib.load(MODELO_PATH)


def predecir(paquete: dict, df_actual: pd.DataFrame) -> pd.DataFrame:
    feature_cols = paquete["feature_cols"]
    faltantes = [c for c in feature_cols if c not in df_actual.columns]
    if faltantes:
        raise RuntimeError(f"Faltan columnas requeridas por el modelo: {faltantes}")

    X = df_actual[feature_cols].fillna(0)
    probas = paquete["model"].predict_proba(X)[:, 1]

    df_actual = df_actual.copy()
    df_actual["probabilidad"] = probas
    df_actual["alerta"] = (probas >= paquete["threshold"]).astype(int)
    return df_actual


# ── Construcción de resultados ────────────────────────────────────────────────
def construir_resultados(df_pred: pd.DataFrame, lunes_actual: pd.Timestamp) -> pd.DataFrame:
    filas = []
    for _, fila in df_pred.iterrows():
        zona = fila["zona"]
        filas.append({
            "zona": zona,
            "clientes": ", ".join(ZONA_CLIENTE.get(zona, [])),
            "semana_scoring": lunes_actual.date(),
            "horizonte_dias": HORIZONTE_DIAS,
            "probabilidad": round(float(fila["probabilidad"]), 4),
            "alerta": int(fila["alerta"]),
            "modelo_version": MODELO_VERSION,
        })
    return pd.DataFrame(filas)


# ── Escritura a Postgres ──────────────────────────────────────────────────────
def escribir_postgres(df: pd.DataFrame) -> None:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL no está configurado. Define la cadena de conexión de Neon en .env "
            "(ver .env.example) antes de escribir resultados en Postgres."
        )
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        for _, fila in df.iterrows():
            # scored_at se calcula en Python (no NOW(), que es específico de Postgres)
            # para que el mismo INSERT funcione contra Neon y contra SQLite local.
            datos = fila.to_dict()
            datos["scored_at"] = datetime.now(timezone.utc)
            conn.execute(
                text("""
                    INSERT INTO riesgo_zona_semana
                        (zona, clientes, semana_scoring, horizonte_dias, probabilidad, alerta, modelo_version, scored_at)
                    VALUES (:zona, :clientes, :semana_scoring, :horizonte_dias, :probabilidad, :alerta, :modelo_version, :scored_at)
                    ON CONFLICT (zona, semana_scoring, horizonte_dias)
                    DO UPDATE SET
                        probabilidad = excluded.probabilidad,
                        alerta = excluded.alerta,
                        modelo_version = excluded.modelo_version,
                        scored_at = excluded.scored_at
                """),
                datos,
            )


# ── Reporte ───────────────────────────────────────────────────────────────────
def _imprimir_resumen(df: pd.DataFrame, lunes_actual: pd.Timestamp) -> None:
    print(f"\n[{lunes_actual.date()}] Scoring completado — {len(df)} zonas evaluadas\n")
    print(f"{'Zona':<18}│{'Clientes':<10}│{'P(y)':<7}│ Alerta")
    print("-" * 50)
    for _, fila in df.iterrows():
        n_clientes = len(fila["clientes"].split(", ")) if fila["clientes"] else 0
        marca = "🔴 SÍ" if fila["alerta"] else "⚪ NO"
        print(f"{fila['zona']:<18}│{n_clientes:<10}│{fila['probabilidad']:<7}│ {marca}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    lunes_actual = _semana_actual()

    paquete = cargar_modelo()
    df_actual = calcular_features_actuales(lunes_actual)
    df_pred = predecir(paquete, df_actual)
    df_resultados = construir_resultados(df_pred, lunes_actual)

    # Se imprime antes de escribir a Neon para que el resumen sea visible
    # incluso si la conexión a Postgres todavía no está configurada.
    _imprimir_resumen(df_resultados, lunes_actual)

    escribir_postgres(df_resultados)
    print(f"\n✓ {len(df_resultados)} filas escritas en riesgo_zona_semana ({DATABASE_URL.split(':')[0]})")


if __name__ == "__main__":
    main()
