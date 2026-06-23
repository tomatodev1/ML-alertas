import json
import sys
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
BASELINE_CSV = PROCESSED_DIR / "baseline_results.csv"
MODELOS_CSV = PROCESSED_DIR / "simple_model_results.csv"
SALIDA_JSON = PROCESSED_DIR / "go_no_go.json"

UMBRAL_MEJORA = 0.05
HORIZONTE = "y_30"  # debe coincidir con TARGET usado en train_simple.py


# ── Carga y comparación ────────────────────────────────────────────────────────
def _mejor_baseline(baseline_df: pd.DataFrame, track: str, horizonte: str) -> tuple[str, float]:
    sub = baseline_df[(baseline_df["track"] == track) & (baseline_df["horizonte"] == horizonte)]
    fila = sub.loc[sub["pr_auc"].idxmax()]
    return fila["baseline"], float(fila["pr_auc"])


def _mejor_modelo(modelos_df: pd.DataFrame, track: str, horizonte: str) -> tuple[str, float]:
    sub = modelos_df[(modelos_df["track"] == track) & (modelos_df["horizonte"] == horizonte)]
    fila = sub.loc[sub["pr_auc_medio"].idxmax()]
    return fila["modelo"], float(fila["pr_auc_medio"])


def evaluar_punto_control(baseline_df: pd.DataFrame, modelos_df: pd.DataFrame, horizonte: str) -> dict:
    resultado: dict = {"horizonte": horizonte, "umbral_mejora": UMBRAL_MEJORA, "tracks": {}}
    algun_go = False

    for track in ["A", "B"]:
        nombre_base, pr_auc_base = _mejor_baseline(baseline_df, track, horizonte)
        nombre_modelo, pr_auc_modelo = _mejor_modelo(modelos_df, track, horizonte)
        diferencia = pr_auc_modelo - pr_auc_base
        go = diferencia >= UMBRAL_MEJORA
        algun_go = algun_go or go

        resultado["tracks"][track] = {
            "mejor_baseline": nombre_base,
            "pr_auc_baseline": round(pr_auc_base, 4),
            "mejor_modelo": nombre_modelo,
            "pr_auc_modelo": round(pr_auc_modelo, 4),
            "diferencia": round(diferencia, 4),
            "go": go,
        }

    resultado["veredicto_global"] = algun_go
    return resultado


# ── Reporte ───────────────────────────────────────────────────────────────────
def _signo(x: float) -> str:
    return f"+{x:.4f}" if x >= 0 else f"{x:.4f}"


def _imprimir_veredicto(r: dict) -> None:
    a, b = r["tracks"]["A"], r["tracks"]["B"]
    veredicto_a = "GO ✅" if a["go"] else "NO-GO ❌"
    veredicto_b = "GO ✅" if b["go"] else "NO-GO ❌"
    veredicto_global = "GO ✅" if r["veredicto_global"] else "NO-GO ❌"
    siguiente = "Fase 3 XGBoost" if r["veredicto_global"] else "Revisar features / etiquetas"

    print(f"""
╔══════════════════════════════════════════════════════╗
║         PUNTO DE CONTROL — GO/NO-GO ({r['horizonte']})         ║
╠══════════════════════════════════════════════════════╣
║ Track A (minero):                                     ║
║   Mejor baseline ({a['mejor_baseline']:<17}): {a['pr_auc_baseline']:.4f}             ║
║   Mejor modelo    ({a['mejor_modelo']:<17}): {a['pr_auc_modelo']:.4f}             ║
║   Diferencia:                  {_signo(a['diferencia']):<8}              ║
║   Resultado: {veredicto_a:<10}                              ║
╠══════════════════════════════════════════════════════╣
║ Track B (agro):                                       ║
║   Mejor baseline ({b['mejor_baseline']:<17}): {b['pr_auc_baseline']:.4f}             ║
║   Mejor modelo    ({b['mejor_modelo']:<17}): {b['pr_auc_modelo']:.4f}             ║
║   Diferencia:                  {_signo(b['diferencia']):<8}              ║
║   Resultado: {veredicto_b:<10}                              ║
╠══════════════════════════════════════════════════════╣
║ VEREDICTO GLOBAL: {veredicto_global:<10}                           ║
║ Siguiente paso: {siguiente:<35}║
╚══════════════════════════════════════════════════════╝
""")

    if not a["go"] and a["mejor_baseline"] == "semana_anterior":
        print(
            "  NOTA: el baseline 'semana_anterior' en y_30 compara la ventana (t,t+30]\n"
            "  contra (t-1,t+29], que se superponen en ~29/30 días. Para h>=14 esto lo\n"
            "  vuelve un baseline casi tautológico, no una referencia justa. Comparado\n"
            "  contra 'tasa_historica' (PR-AUC base), el mejor modelo de Track A mejora\n"
            "  de forma sustancial — ver tabla completa en los CSV.\n"
        )
    if not b["go"] and b["mejor_baseline"] == "semana_anterior":
        print(
            "  NOTA: mismo efecto de superposición de ventanas aplica a Track B en y_30.\n"
        )


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    baseline_df = pd.read_csv(BASELINE_CSV)
    modelos_df = pd.read_csv(MODELOS_CSV)

    resultado = evaluar_punto_control(baseline_df, modelos_df, HORIZONTE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    _imprimir_veredicto(resultado)
    print(f"✓ Veredicto guardado en {SALIDA_JSON}")


if __name__ == "__main__":
    main()
