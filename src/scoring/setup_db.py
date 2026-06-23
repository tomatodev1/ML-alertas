"""Crea (si no existe) la tabla riesgo_zona_semana en el backend configurado
en DATABASE_URL (Neon/Postgres en producción; SQLite local para pruebas)."""
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

DDL_POSTGRES = """
    CREATE TABLE IF NOT EXISTS riesgo_zona_semana (
        id                SERIAL PRIMARY KEY,
        zona              TEXT NOT NULL,
        clientes          TEXT NOT NULL,
        semana_scoring    DATE NOT NULL,
        horizonte_dias    INTEGER NOT NULL,
        probabilidad      FLOAT NOT NULL,
        alerta            INTEGER NOT NULL,
        modelo_version    TEXT NOT NULL,
        scored_at         TIMESTAMP DEFAULT NOW(),
        UNIQUE(zona, semana_scoring, horizonte_dias)
    )
"""

DDL_SQLITE = """
    CREATE TABLE IF NOT EXISTS riesgo_zona_semana (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        zona              TEXT NOT NULL,
        clientes          TEXT NOT NULL,
        semana_scoring    DATE NOT NULL,
        horizonte_dias    INTEGER NOT NULL,
        probabilidad      FLOAT NOT NULL,
        alerta            INTEGER NOT NULL,
        modelo_version    TEXT NOT NULL,
        scored_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(zona, semana_scoring, horizonte_dias)
    )
"""

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_riesgo_zona ON riesgo_zona_semana(zona)",
    "CREATE INDEX IF NOT EXISTS idx_riesgo_semana ON riesgo_zona_semana(semana_scoring)",
    "CREATE INDEX IF NOT EXISTS idx_riesgo_alerta ON riesgo_zona_semana(alerta)",
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL no está configurado. Copia .env.example a .env y pon ahí "
            "la cadena de conexión de tu proyecto Neon (o sqlite:///data/archivo.db para pruebas)."
        )

    engine = create_engine(DATABASE_URL)
    ddl_tabla = DDL_SQLITE if engine.dialect.name == "sqlite" else DDL_POSTGRES

    with engine.begin() as conn:
        conn.execute(text(ddl_tabla))
        for statement in INDICES:
            conn.execute(text(statement))

    print(f"✓ Tabla riesgo_zona_semana creada (o ya existía) en {engine.dialect.name}, con sus 3 índices.")


if __name__ == "__main__":
    main()
