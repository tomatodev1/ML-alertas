import re
import sys
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ARCHIVO = Path("src/data") / "BD_Incidentes - copia.xlsx"
HOJA = "INCIDENTES"
INTERIM_DIR = Path("data/interim")
SALIDA = INTERIM_DIR / "incidentes_normalizados.parquet"

# ── Mapeo MOTIVO → categoría (mismo esquema que loader_alertas.py) ───────────
# PROTESTA y VIOLENCIA están dentro de alcance (CLAUDE.md: "paros, bloqueos,
# protestas, coyuntura política, inseguridad"). Accidentes/desastres naturales
# quedan en OTRO (no son conflicto social) y se excluyen de los features.
CATEGORIA_MAP = {
    "PROTESTA": "PROTESTA",
    "PROTESTA / PESCA": "PROTESTA",
    "POSIBLE PROTESTA": "PROTESTA",
    "BLOQUEO": "PROTESTA",
    "RECLAMO": "PROTESTA",
    "REUNIÓN": "PROTESTA",
    "EVENTO SOCIAL": "PROTESTA",
    "SOCIAL": "PROTESTA",
    "INCIDENTE FUNDO": "PROTESTA",
    "INCIDENTE MINA": "PROTESTA",
    "MOVIMIENTO POLITICO": "POLITICA",
    "MOVIMIENTO POLÍTICO": "POLITICA",
    "POLITICO ELECTORAL": "POLITICA",
    "HOMICIDIO": "VIOLENCIA",
    "SICARIATO": "VIOLENCIA",
    "SECUESTRO": "VIOLENCIA",
    "EXTORSIÓN": "VIOLENCIA",
    "EXTORCION": "VIOLENCIA",
    "BANDA CRIMINAL": "VIOLENCIA",
    "VIOLENCIA CIUDADANA": "VIOLENCIA",
    "DETENCIÓN": "VIOLENCIA",
    "ROBO": "VIOLENCIA",
}
# accidente vehicular, desastre natural, obras, presencia policial, minería → OTRO

# ── Mapeo departamento (+ provincia) → zona canónica del proyecto ────────────
DEPTO_MAP = {
    "ANCASH": "Áncash",
    "CAJAMARCA": "Cajamarca",
    "ICA": "Ica",
    "LA LIBERTAD": "La Libertad",
    "HUANUCO": "Huánuco",
    "HUÁNUCO": "Huánuco",
    "PASCO": "Pasco",
}
# Lima Metropolitana queda fuera de alcance; solo Huaral (parte de
# "Supe / Huaura / Huaral" / Lima Provincias) se reasigna explícitamente.
PROVINCIA_A_ZONA = {
    "HUARAL": "Lima Provincias",
}
# Fuera de alcance explícito (Loreto/Petrotal) o fuera de las 10 zonas del proyecto:
# Loreto, Piura, Arequipa, Lambayeque, Huancavelica, Junín, etc. → se descartan.


def _norm(valor) -> str:
    return re.sub(r"\s+", " ", str(valor).strip().upper())


def _resolver_zona(depto_norm: str, provincia_norm: str) -> str | None:
    if provincia_norm in PROVINCIA_A_ZONA:
        return PROVINCIA_A_ZONA[provincia_norm]
    return DEPTO_MAP.get(depto_norm)


# ── Loader principal ───────────────────────────────────────────────────────────
def cargar_incidentes() -> pd.DataFrame:
    raw = pd.read_excel(ARCHIVO, sheet_name=HOJA, header=1)

    df = pd.DataFrame()
    df["fecha"] = pd.to_datetime(raw["ID_FECHA"], errors="coerce")

    depto_norm = raw["ID_DEPARTAMENTO"].apply(_norm)
    provincia_norm = raw["ID_PROVINCIA"].apply(_norm)
    df["zona"] = [
        _resolver_zona(d, p) for d, p in zip(depto_norm, provincia_norm)
    ]

    df["provincia"] = raw["ID_PROVINCIA"].astype(str).str.strip()
    df["distrito"] = raw["DISTRITO"].astype(str).str.strip()
    df["categoria_original"] = raw["MOTIVO"].astype(str).str.strip()
    df["categoria"] = raw["MOTIVO"].apply(_norm).map(CATEGORIA_MAP).fillna("OTRO")
    df["titulo"] = raw["TITULO"].astype(str).str.strip()
    df["fuente"] = "bd_incidentes_interna"

    n_antes = len(df)
    df = df[df["fecha"].notna() & df["zona"].notna()].copy()
    n_fuera_zona = n_antes - len(df)

    n_antes_dedup = len(df)
    df = df.drop_duplicates(subset=["fecha", "zona", "titulo"])
    n_duplicados = n_antes_dedup - len(df)

    if n_fuera_zona:
        print(f"  ⚠ Descartadas {n_fuera_zona} filas sin fecha válida o fuera de las 10 zonas del proyecto.")
    if n_duplicados:
        print(f"  ⚠ Descartados {n_duplicados} duplicados exactos (fecha+zona+título).")

    return df


# ── Guardar ───────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)


# ── Resumen ───────────────────────────────────────────────────────────────────
def _resumen(df: pd.DataFrame) -> None:
    rango_min = df["fecha"].min().date()
    rango_max = df["fecha"].max().date()

    zonas = df["zona"].value_counts()
    zonas_str = ", ".join(f"{z}({n})" for z, n in zonas.items())

    categorias = df["categoria"].value_counts()
    categorias_str = ", ".join(f"{c}({n})" for c, n in categorias.items())

    print(f"✓ Incidentes normalizados: {len(df):,} filas")
    print(f"  Rango: {rango_min} → {rango_max}")
    print(f"  Zonas: {zonas_str}")
    print(f"  Categorías: {categorias_str}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar_incidentes()
    guardar(df)
    _resumen(df)


if __name__ == "__main__":
    main()
