"""Test mínimo de integración end-to-end del pipeline de scoring semanal."""
import sys
from datetime import date

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.scoring.score_semanal import (
    DATABASE_URL,
    _semana_actual,
    cargar_modelo,
    calcular_features_actuales,
    predecir,
)

load_dotenv()

ZONA_TEST = "__TEST__"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("1. Cargando modelo...")
    paquete = cargar_modelo()
    assert "model" in paquete and "threshold" in paquete
    print("   OK")

    print("2. Calculando features de la semana actual...")
    lunes_actual = _semana_actual()
    df_actual = calcular_features_actuales(lunes_actual)
    columnas_criticas = ["n_total_1w", "tasa_pobreza", "mes", "trimestre"]
    nans = df_actual[columnas_criticas].isna().sum()
    assert nans.sum() == 0, f"NaN en columnas críticas: {nans[nans > 0].to_dict()}"
    print(f"   OK ({len(df_actual)} zonas, sin NaN en columnas críticas)")

    print("3. Prediciendo...")
    df_pred = predecir(paquete, df_actual)
    assert df_pred["probabilidad"].between(0, 1).all()
    print("   OK (probabilidades entre 0 y 1)")

    print("4. Conectando a la base de datos (DATABASE_URL)...")
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL no está configurado. Copia .env.example a .env y pon ahí "
            "la cadena de conexión de tu proyecto Neon."
        )
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("   OK")

    print("5. Escribiendo fila de test...")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO riesgo_zona_semana
                    (zona, clientes, semana_scoring, horizonte_dias, probabilidad, alerta, modelo_version)
                VALUES (:zona, 'test', :semana, 30, 0.5, 0, 'test')
                ON CONFLICT (zona, semana_scoring, horizonte_dias)
                DO UPDATE SET probabilidad = EXCLUDED.probabilidad
            """),
            {"zona": ZONA_TEST, "semana": date.today()},
        )
    print("   OK")

    print("6. Leyendo fila de test...")
    with engine.connect() as conn:
        resultado = conn.execute(
            text("SELECT * FROM riesgo_zona_semana WHERE zona = :zona"), {"zona": ZONA_TEST}
        ).fetchone()
    assert resultado is not None
    print("   OK")

    print("7. Borrando fila de test...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM riesgo_zona_semana WHERE zona = :zona"), {"zona": ZONA_TEST})
    print("   OK")

    print("\n✓ Pipeline de scoring funcional end-to-end")


if __name__ == "__main__":
    main()
