import re
import sys
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
SRC_DIR = Path("src/data")
INTERIM_DIR = Path("data/interim")


def _encontrar_excel() -> Path:
    """Busca el FRM01 más reciente en src/data/ (el nombre varía por versión)."""
    candidatos = sorted(SRC_DIR.glob("FRM01_Formulario_informacion*.xlsx"))
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró FRM01_Formulario_informacion*.xlsx en {SRC_DIR}"
        )
    return candidatos[-1]


# ── Mapeos ────────────────────────────────────────────────────────────────────
NIVEL_MAP = {
    "alto": "CRIT",
    "medio": "ALRT",
    "bajo": "INFO",
}

CATEGORIA_MAP = {
    # PROTESTA
    "protestas, paros y bloqueos": "PROTESTA",   # variante más frecuente en datos (coma)
    "protestas paros y bloqueos": "PROTESTA",
    "protestas-paros-bloqueos": "PROTESTA",
    "protestas-paros y bloqueos": "PROTESTA",
    "protestas paros y bloqueos-pnp": "PROTESTA",
    "protestas sociales": "PROTESTA",
    "protestas-paros-y-bloqueos": "PROTESTA",
    "paro-transportistas-combustible": "PROTESTA",
    "agricultores-paro-pcm": "PROTESTA",
    "social y transporte": "PROTESTA",
    "seguridad-protestas": "PROTESTA",
    "seguridad y conflictos": "PROTESTA",
    # VIOLENCIA
    "seguridad y delincuencia": "VIOLENCIA",
    "seguridad interna": "VIOLENCIA",
    "policial": "VIOLENCIA",
    "accidente de transito": "VIOLENCIA",
    "mineriailegal-accidentes-trabajo": "VIOLENCIA",
    "fenomenos-naturales": "VIOLENCIA",
    "sismo-igp-prevención": "VIOLENCIA",
    # POLITICA
    "coyuntura política": "POLITICA",
    "coyuntura politica": "POLITICA",
    "coyuntura social": "POLITICA",
    "corrupción municipal": "POLITICA",
    "politica-jne-alvarohenzler": "POLITICA",
    "politica-constitucion-referendum": "POLITICA",
    "mineriailegal-confiep-politica": "POLITICA",
    # ECONOMIA
    "economía-desarrollo": "ECONOMIA",
    "economía - desarrollo": "ECONOMIA",
    "economia-desarrollo": "ECONOMIA",
    "economia-costos": "ECONOMIA",
    "agroexportacion": "ECONOMIA",
    "energía y recursos": "ECONOMIA",
    "energia y combustibles": "ECONOMIA",
    "minería y energía": "ECONOMIA",
    "minería-formalización": "ECONOMIA",
    "cafe-exportaciones-economia": "ECONOMIA",
    "agro-agricultores-inia": "ECONOMIA",
    "antamina-sostenibilidad-snmpe": "ECONOMIA",
    "marcobre-fitch": "ECONOMIA",
    "salud": "ECONOMIA",
    # AMBIENTAL
    "agro-agua-medioambiente": "AMBIENTAL",
    "mineriailegal-rioscontaminados-hualgayoc-bambamarca": "AMBIENTAL",
}
# Todo lo que no esté en el mapa → "OTRO"

ZONA_MAP = {
    "ANCASH": "Áncash",
    "ANCCASH": "Áncash",
    "HUANUCO": "Huánuco",
    "HUÁNUCO": "Huánuco",
    "LA LIBERTAD": "La Libertad",
    "LALIBERTAD": "La Libertad",
    "LIBERTAD": "La Libertad",
    "CAJAMARCA": "Cajamarca",
    "PASCO": "Pasco",
    "CERRO DE PASCO": "Pasco",
    "ICA": "Ica",
    "LIMA": "Lima Provincias",
    "LIMA PROVINCIAS": "Lima Provincias",
    "PISCO": "Pisco",
    "HUAURA": "Huaura",
    "NACIONAL": "NACIONAL",
    # Resto (LORETO, AREQUIPA, etc.) → se deja el valor original para revisión
}


# ── Normalización de strings ───────────────────────────────────────────────────
def _norm_cat(valor: str) -> str:
    """Lowercase, strip y colapsa espacios múltiples."""
    return re.sub(r"\s+", " ", str(valor).lower().strip())


def _norm_zona(valor: str) -> str:
    return str(valor).upper().strip()


def _encontrar_col(df: pd.DataFrame, prefijo: str) -> str:
    """Devuelve el nombre de la primera columna que empiece con el prefijo dado."""
    for col in df.columns:
        if col.startswith(prefijo):
            return col
    raise KeyError(f"No se encontró columna con prefijo '{prefijo}'. Columnas: {list(df.columns)}")


# ── Loader principal ───────────────────────────────────────────────────────────
def cargar_alertas() -> pd.DataFrame:
    ruta = _encontrar_excel()
    raw = pd.read_excel(ruta)

    # Nombres reales de columnas ambiguas (Relevancia tiene nombre larguísimo;
    # "Insertar noticia" tiene un typo en el archivo fuente).
    col_relevancia = _encontrar_col(raw, "Relevancia")
    col_resumen = _encontrar_col(raw, "Insertar noticia")

    df = pd.DataFrame()

    # fecha
    df["fecha"] = pd.to_datetime(raw["Marca temporal"], dayfirst=False, errors="coerce")

    # zona: uppercase + strip → ZONA_MAP; si no está, dejar valor original
    zona_norm = raw["ID_Departamento"].apply(_norm_zona)
    df["zona"] = zona_norm.map(ZONA_MAP).fillna(zona_norm)

    df["provincia"] = raw["Provincia"].astype(str).str.strip()

    # nivel: lowercase + strip → NIVEL_MAP; si no está, NaN (se filtrará o marcará)
    df["nivel"] = (
        raw[col_relevancia]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(NIVEL_MAP)
    )

    # categoria
    df["categoria_original"] = raw["Categoria"].astype(str).str.strip()
    df["categoria"] = (
        df["categoria_original"]
        .apply(_norm_cat)
        .map(CATEGORIA_MAP)
        .fillna("OTRO")
    )

    df["cliente"] = raw["Cliente"].astype(str).str.strip()
    df["resumen"] = raw[col_resumen].astype(str).str.strip()
    df["etiquetas"] = raw["Etiquetas"].astype(str).str.strip()
    df["fuente"] = "alertas_propias"

    # Filtrar filas sin fecha o sin zona
    n_antes = len(df)
    df = df[df["fecha"].notna() & df["zona"].notna()].copy()
    n_descartadas = n_antes - len(df)
    if n_descartadas:
        print(f"  ⚠ Descartadas {n_descartadas} filas sin fecha o zona válida.")

    return df


# ── Guardar ───────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    salida_parquet = INTERIM_DIR / "alertas_normalizadas.parquet"
    df.to_parquet(salida_parquet, index=False)

    # CSV de revisión para categorías sin mapear
    otros = df[df["categoria"] == "OTRO"][
        ["fecha", "zona", "categoria_original", "resumen"]
    ].copy()
    salida_otros = INTERIM_DIR / "categorias_sin_mapear.csv"
    otros.to_csv(salida_otros, index=False, encoding="utf-8")


def _resumen(df: pd.DataFrame) -> None:
    rango_min = df["fecha"].min().date()
    rango_max = df["fecha"].max().date()

    zonas = df["zona"].value_counts()
    zonas_str = ", ".join(f"{z}({n})" for z, n in zonas.items())

    niveles = df["nivel"].value_counts()
    niveles_str = ", ".join(f"{n}({c})" for n, c in niveles.items())

    n_otro = (df["categoria"] == "OTRO").sum()

    print(f"✓ Alertas normalizadas: {len(df):,} filas")
    print(f"  Rango: {rango_min} → {rango_max}")
    print(f"  Zonas encontradas: {zonas_str}")
    print(f"  Categorías OTRO: {n_otro} filas (revisar manualmente)")
    print(f"  Niveles: {niveles_str}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar_alertas()
    guardar(df)
    _resumen(df)


if __name__ == "__main__":
    main()
