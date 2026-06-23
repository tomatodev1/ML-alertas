import sys
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ARCHIVO_INEI = Path("src/data") / "INEI TASA DE POBREZA x DEP.xlsx"
INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "inei_pobreza.parquet"

# ── Mapeo depto INEI → zonas canónicas del proyecto ──────────────────────────
# Un departamento puede cubrir varias zonas de predicción.
# Pisco es provincia de Ica; Huarmey es costa de Áncash;
# Barranca/Supe/Huaura/Huaral están en Lima.
ZONA_MAP: dict[str, list[str]] = {
    "Áncash":                ["Áncash", "Huarmey"],
    "Huánuco":               ["Huánuco"],
    "Pasco":                 ["Pasco"],
    "Cajamarca":             ["Cajamarca"],
    "La Libertad":           ["La Libertad"],
    "Ica":                   ["Ica", "Pisco"],
    "Lima (Región + Metro)": ["Barranca", "Lima Provincias"],
}

AÑOS = [2023, 2024, 2025]


# ── Carga ─────────────────────────────────────────────────────────────────────
def cargar_inei() -> pd.DataFrame:
    # Fila 2 (0-indexed) es el encabezado real; columna 0 está vacía en el Excel.
    raw = pd.read_excel(ARCHIVO_INEI, header=2, usecols=[1, 2, 3, 4])
    raw.columns = ["departamento", "pobreza_2023", "pobreza_2024", "pobreza_2025"]

    raw["departamento"] = raw["departamento"].astype(str).str.strip().str.title()
    # Eliminar filas vacías o con strings vacíos tras la conversión
    raw = raw[~raw["departamento"].isin({"Nan", "None", ""})]

    # Expandir a formato largo zona | año | tasa_pobreza
    registros: list[dict] = []
    for _, fila in raw.iterrows():
        zonas = ZONA_MAP.get(fila["departamento"])
        if zonas is None:
            continue  # deptos fuera de scope (Arequipa, Loreto, etc.) y PROMEDIO NACIONAL
        for zona in zonas:
            for año in AÑOS:
                registros.append({
                    "zona": zona,
                    "año": año,
                    "tasa_pobreza": float(fila[f"pobreza_{año}"]),
                })

    return pd.DataFrame(registros)


# ── Guardar ───────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)


# ── Resumen ───────────────────────────────────────────────────────────────────
def _resumen(df: pd.DataFrame) -> None:
    n_zonas = df["zona"].nunique()
    n_años = df["año"].nunique()

    idx_min = df["tasa_pobreza"].idxmin()
    idx_max = df["tasa_pobreza"].idxmax()
    zona_min = df.loc[idx_min, "zona"]
    zona_max = df.loc[idx_max, "zona"]
    val_min = df.loc[idx_min, "tasa_pobreza"]
    val_max = df.loc[idx_max, "tasa_pobreza"]

    print(f"✓ INEI procesado: {n_zonas} zonas × {n_años} años = {len(df)} filas")
    print(f"  Rango pobreza: {val_min}% ({zona_min}) → {val_max}% ({zona_max})")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar_inei()
    guardar(df)
    _resumen(df)


if __name__ == "__main__":
    main()
