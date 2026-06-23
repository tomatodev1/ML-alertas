# Proyecto: Modelo Predictivo de Conflictos Sociales — PROTECTA PERÚ

Sistema de alerta temprana que estima, por zona de operación y con anticipación,
la probabilidad de un conflicto social o protesta. Complementa (no reemplaza) el
sistema de alertas reactivo actual.

## Objetivo

Clasificación binaria supervisada con componente temporal (pronóstico): para una
zona y una ventana futura, predecir si habrá conflicto/protesta (sí/no). El
entregable final es un puntaje de riesgo semanal por zona, escrito a PostgreSQL y
consumido por el sistema de alertas existente.

## Decisiones clave (NO cambiar sin justificar)

- **Etiqueta:** binaria. Definida como "aparición de conflicto nuevo o evento de
  protesta activa" en la ventana — NO la mera existencia de un conflicto crónico
  (eso haría el target trivialmente positivo en zonas como Áncash).
- **Unidad de predicción:** una fila = `zona × semana` (granularidad semanal).
- **Horizontes:** evaluar multi-horizonte `y_7`, `y_14`, `y_30`, `y_60` días.
- **Tipo de conflicto:** CUALQUIER conflicto/protesta que afecte directa o
  indirectamente al cliente (paros, bloqueos, protestas, coyuntura política,
  inseguridad). NO se limita a minería ni agro.
- **Dos tracks de etiqueta** (difieren en la FUENTE del label, no en el tipo de
  conflicto):
  - Track A (departamental, label desde Defensoría): Áncash, Huánuco, Pasco,
    Cajamarca, La Libertad.
  - Track B (sub-departamental, label por eventos ACLED/GDELT + alertas propias):
    Ica, Pisco, Huarmey, Barranca, Supe, Huaura, Huaral.
- **Ruteo zona → cliente:** capa determinista SEPARADA del modelo (evolución del
  `router.py` actual). El modelo predice por zona; el router traduce a cliente(s),
  multi-etiqueta. El riesgo NACIONAL es una feature global, no una zona.
- **Petrotal (Loreto/Ucayali): FUERA de alcance.**

## Zonas y clientes

| Zona | Cliente(s) |
|------|-----------|
| Áncash | Antamina |
| Huánuco | Antamina |
| Pasco | Antamina |
| Cajamarca | Bechtel / Newmont |
| La Libertad | Clientes Norte (híbrido sierra/costa) |
| Ica | Agrícola Chapi, Clientes Sur |
| Pisco | Pisco |
| Huarmey (costa de Áncash) | Agrícola Huarmey |
| Barranca | Agrícola Santa Azul, Agrícola Huarmey |
| Supe / Huaura / Huaral | Agrícola Santa Azul |

## Fuentes de datos

- **Histórico propio (PostgreSQL/Neon):** 3 años de alertas clasificadas por
  depto, nivel (CRIT/ALRT/INFO) y categoría. Features principales.
- **Defensoría del Pueblo:** reporte mensual de conflictos por depto. Etiqueta
  (track A).
- **ACLED:** eventos de protesta geolocalizados. Etiqueta/feature (track B).
  OJO: uso comercial puede requerir licencia; el proyecto debe funcionar sin ella.
- **GDELT:** intensidad/tono mediático de conflicto. Feature.
- **Calendario (ONPE/JNE, feriados, fechas críticas):** features estructurales.
- **INEI:** indicadores socioeconómicos por depto. Features lentas.
- **Commodities (cobre/oro):** feature en zonas mineras.

## Regla crítica: anti-fuga temporal (leakage)

- Toda feature de la semana `t` usa SOLO datos disponibles hasta la fecha de corte
  (fin de la semana t). Nada posterior.
- El label `y_h` mira la ventana futura `[t, t+h]`. Features miran atrás.
- Defensoría se publica con rezago (reporte del mes M sale en M+1): usar SIEMPRE el
  último reporte efectivamente publicado a la fecha de corte, nunca el del mes en
  curso.
- INEI/commodities: valor "as-of" (último publicado a la fecha).

## Stack

- Python + venv. pandas, scikit-learn, xgboost/lightgbm, jupyter.
- Texto: sentence-transformers multilingüe o embeddings ya integrados.
- Fases 0-3: trabajo 100% LOCAL. El histórico de alertas y demás fuentes se
  exportan a archivos (CSV/Excel/Parquet) dentro de `data/` (gitignored). No se
  configura Neon ni infraestructura de despliegue todavía.
- BD (Fase 4, despliegue): PostgreSQL (Neon) + sqlalchemy/psycopg2. Cron job en
  Railway que escribe a Postgres. Se evalúa si realmente se necesita al llegar a
  esa fase.

## Metodología (no negociable)

- **Validación temporal** (walk-forward / TimeSeriesSplit), NUNCA k-fold aleatorio.
- Métricas: **PR-AUC** y **recall a precisión fija**, NO accuracy (clases
  desbalanceadas).
- **Punto de control go/no-go:** el modelo debe superar un baseline trivial
  (predecir lo del periodo anterior o tasa histórica de la zona). Si no, detener.

## Ruta de desarrollo (fase actual: FASE 0)

0. **Setup mínimo + definición** — repo, venv, requirements, conexión a Neon,
   definición escrita de etiqueta/horizontes. NO montar infra pesada aún.
1. **Dataset maestro** (`zona × semana`) — ETL + integración de fuentes + features
   + etiquetas, respetando anti-fuga. ES EL CORAZÓN DEL PROYECTO.
2. **Baseline + PoC** — baseline tonto, modelo simple, validación temporal →
   PUNTO DE CONTROL.
3. **Modelo** — XGBoost/LightGBM, desbalance, ajuste de horizontes, guardar `.pkl`.
4. **Despliegue** — script de scoring semanal → tabla `riesgo_zona_semana` en
   Postgres → integrar al flujo de alertas.
5. **Monitoreo** — comparar predicción vs realidad, reentrenar mensual.

REGLA DE ORO: no escribir código de modelo hasta que la tabla maestra (Fase 1)
esté lista.

## Estructura de repositorio

```
prediccion-conflictos/
├── requirements.txt
├── notebooks/            # exploración (Fases 1-3)
├── src/
│   ├── data/             # extracción + construcción del dataset
│   ├── features/         # cálculo de features (reutilizable)
│   ├── models/           # entrenamiento y evaluación
│   └── scoring/          # predicción semanal (Fase 4)
├── models/               # artefactos entrenados (.pkl)
└── data/                 # datasets intermedios
```

## Convenciones

- Código y comentarios pueden ir en español.
- Las fechas de corte y ventanas SIEMPRE explícitas en el código (evitar fuga).
- Antes de proponer un modelo, confirmar que el baseline existe y está medido.

## Estado actual (junio 2026)

### Decisiones de infraestructura
- Base de datos: SQLite local (`data/riesgo_zona_semana.db`)
- Despliegue: NO — el sistema corre localmente, sin Railway ni Neon
- Acceso: solo el analista de datos

### Operación semanal
Cada lunes ejecutar manualmente:
```bash
python src/scoring/score_semanal.py   # calcula riesgo 5 zonas
python src/scoring/query_riesgo.py    # muestra tabla de resultados
```

### Modelo activo (Track B)
- Archivo: models/modelo_v1_track_B.pkl
- Algoritmo: LightGBM, horizonte y_30, umbral=0.5
- PR-AUC: 0.714 | Tasa de alerta esperada: 16.7%
- Zonas: Ica, Pisco, Huarmey, Barranca, Lima Provincias

### Track A (minero) — pausado
Reactivar cuando haya más historia o fuente de etiqueta alternativa.
Zonas pendientes: Áncash, Huánuco, Pasco, Cajamarca, La Libertad

### Fase 5 — Validación operativa
Cada viernes anotar en una hoja simple:
¿Hubo conflicto/protesta real esta semana en cada zona? (sí/no)
Con 6 semanas de datos ya puedes calcular el recall operativo real.