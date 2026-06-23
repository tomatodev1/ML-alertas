"""Uso: python -m src.scoring.query_riesgo [--zona Ica] [--semanas 4]"""
import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def consultar(zona: str | None, semanas: int) -> pd.DataFrame:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL no está configurado. Copia .env.example a .env y pon ahí "
            "la cadena de conexión de tu proyecto Neon."
        )
    engine = create_engine(DATABASE_URL)
    fecha_limite = (pd.Timestamp.today().normalize() - pd.Timedelta(weeks=semanas)).date()

    sql = """
        SELECT semana_scoring, zona, clientes, probabilidad, alerta
        FROM riesgo_zona_semana
        WHERE semana_scoring >= :fecha_limite
    """
    params: dict = {"fecha_limite": fecha_limite}
    if zona:
        sql += " AND zona = :zona"
        params["zona"] = zona
    sql += " ORDER BY semana_scoring, zona"

    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def _imprimir(df: pd.DataFrame) -> None:
    if df.empty:
        print("Sin resultados para los filtros indicados.")
        return
    print(f"{'Semana':<14}│{'Zona':<18}│{'Clientes':<30}│{'P(y)':<7}│Alerta")
    print("-" * 85)
    for _, fila in df.iterrows():
        marca = "🔴" if fila["alerta"] else "⚪"
        print(f"{str(fila['semana_scoring']):<14}│{fila['zona']:<18}│{fila['clientes']:<30}│{fila['probabilidad']:<7}│{marca}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Consulta el estado actual de riesgo_zona_semana")
    parser.add_argument("--zona", default=None, help="Filtrar por una zona específica")
    parser.add_argument("--semanas", type=int, default=4, help="Número de semanas recientes a mostrar")
    args = parser.parse_args()

    df = consultar(args.zona, args.semanas)
    _imprimir(df)


if __name__ == "__main__":
    main()
