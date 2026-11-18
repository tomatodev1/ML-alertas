"""Calibración de umbrales ALTO / MEDIO — Track A Áncash, sobre predicciones OOF.

Regla (v2, 2026-11-18), con probabilidades calibradas (LR sin balanceo):
  - ALTO  = menor umbral (rejilla de 0.01) tal que la precisión OOF de
            "alerta si p >= umbral" es >= PRECISION_ALTO para ese umbral Y
            para todos los superiores (evita picos inestables).
  - MEDIO = tasa base de protesta en las filas OOF (redondeada a 0.01):
            "probabilidad por encima del promedio histórico". Es la misma
            semántica que tenía v1 (MEDIO = 44% = tasa histórica de entonces).
  - BAJO  = por debajo de MEDIO.

Los umbrales se guardan en models/umbrales_ancash.json y los leen el scoring
y el export del dashboard. Se recalculan SOLO al correr este módulo (o
`export_dashboard.py --recalibrar`): los textos del dashboard dependen de
ellos, no deben moverse cada semana.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import evaluacion_ancash as ev

OOF_PATH = Path("data/processed") / "oof_ancash.parquet"
UMBRALES_PATH = Path("models") / "umbrales_ancash.json"
TABLA_PATH = Path("data/processed") / "calibracion_umbral_ancash.csv"

PRECISION_ALTO = 0.80
REJILLA = np.round(np.arange(0.30, 0.96, 0.01), 2)


def barrer(y: pd.Series, p: pd.Series) -> pd.DataFrame:
    filas = []
    for t in REJILLA:
        prec, rec, tasa = ev.precision_recall_acumulado(y, p, t)
        filas.append({"umbral": t, "precision": prec, "recall": rec, "tasa_alerta": tasa})
    return pd.DataFrame(filas)


VENTANA_ESTABILIDAD = 0.10   # la precisión debe sostenerse en [t, t + 0.10]
HISTERESIS = 0.02            # cambios menores a esto no mueven el umbral vigente
UMBRALES_V1 = {"umbral_alto": 0.80, "umbral_medio": 0.44}


def _vigentes() -> dict:
    try:
        return cargar()
    except (FileNotFoundError, ValueError):
        return dict(UMBRALES_V1)


def calibrar(y: pd.Series, p: pd.Series, vigentes: dict | None = None) -> dict:
    tabla = barrer(y, p)
    prec = tabla.set_index("umbral")["precision"].fillna(0)
    alto = float(REJILLA.max())
    for t in REJILLA:
        ventana = prec[(prec.index >= t - 1e-9) & (prec.index <= t + VENTANA_ESTABILIDAD + 1e-9)]
        if (ventana >= PRECISION_ALTO).all():
            alto = float(t)
            break
    medio = round(float(y[p.notna()].mean()), 2)

    vig = vigentes if vigentes is not None else _vigentes()
    nota = []
    if abs(alto - vig["umbral_alto"]) <= HISTERESIS + 1e-9:
        nota.append(f"ALTO calculado {alto:.2f} ≈ vigente {vig['umbral_alto']:.2f} (histéresis) → se mantiene")
        alto = vig["umbral_alto"]
    if abs(medio - vig["umbral_medio"]) <= HISTERESIS + 1e-9:
        nota.append(f"MEDIO calculado {medio:.2f} ≈ vigente {vig['umbral_medio']:.2f} (histéresis) → se mantiene")
        medio = vig["umbral_medio"]
    if medio >= alto:
        medio = round(alto - 0.05, 2)
    return {"umbral_alto": round(alto, 2), "umbral_medio": round(medio, 2), "tabla": tabla,
            "notas": nota,
            "regla": (f"ALTO: menor umbral cuya precisión OOF acumulada se mantiene >= {PRECISION_ALTO} "
                      f"en [t, t+{VENTANA_ESTABILIDAD}]; MEDIO: tasa base OOF de protesta; "
                      f"histéresis ±{HISTERESIS} respecto al umbral vigente")}


def guardar(umbrales: dict, n_oof: int) -> None:
    UMBRALES_PATH.parent.mkdir(parents=True, exist_ok=True)
    UMBRALES_PATH.write_text(json.dumps({
        "umbral_alto": umbrales["umbral_alto"],
        "umbral_medio": umbrales["umbral_medio"],
        "n_niveles": 3,
        "regla": umbrales["regla"],
        "n_oof": n_oof,
        "fecha_calibracion": str(pd.Timestamp.now().date()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def cargar() -> dict:
    return json.loads(UMBRALES_PATH.read_text(encoding="utf-8"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    oof = pd.read_parquet(OOF_PATH)
    y, p = oof["y_30"].astype(int), oof["prob"]
    vig = _vigentes()
    u = calibrar(y, p, vig)
    for n in u["notas"]:
        print("  ·", n)
    if (u["umbral_alto"], u["umbral_medio"]) != (vig["umbral_alto"], vig["umbral_medio"]):
        print(f"  ⚠ CAMBIO DE UMBRALES: ALTO {vig['umbral_alto']:.2f} → {u['umbral_alto']:.2f} · "
              f"MEDIO {vig['umbral_medio']:.2f} → {u['umbral_medio']:.2f} (actualizar textos del dashboard)")
    u["tabla"].to_csv(TABLA_PATH, index=False)
    guardar(u, len(oof))
    print(f"Umbrales calibrados sobre {len(oof)} predicciones OOF: ALTO >= {u['umbral_alto']:.2f} · "
          f"MEDIO >= {u['umbral_medio']:.2f}")
    print(ev.metricas_por_nivel(y, p, u["umbral_alto"], u["umbral_medio"]).to_string(index=False))
    print(f"✓ Guardado en {UMBRALES_PATH}")


if __name__ == "__main__":
    main()
