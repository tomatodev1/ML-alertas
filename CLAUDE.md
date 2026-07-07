 # Proyecto: Modelo Predictivo de Conflictos Sociales — Áncash (ANTAMINA)

Sistema de alerta temprana que estima, por Unidad de Gestión Territorial (UGT)
de ANTAMINA en Áncash y con anticipación, la probabilidad de un conflicto
social o protesta. Complementa (no reemplaza) el sistema de alertas reactivo
actual.

**Alcance: exclusivamente la región de Áncash y el cliente ANTAMINA.** El
proyecto se reenfocó a este alcance único; no se mantiene código, datos ni
configuración de otras zonas o clientes.

## Objetivo

Clasificación binaria supervisada con componente temporal (pronóstico): para
una UGT y una ventana futura, predecir si habrá conflicto/protesta (sí/no). El
entregable es un puntaje de riesgo semanal por UGT, consumido vía el dashboard
predictivo y el script de scoring.

## Decisiones clave (NO cambiar sin justificar)

- **Etiqueta:** binaria. Definida como "protesta nueva en la UGT" en la
  ventana futura — NO la mera existencia de conflicto crónico (eso haría el
  target trivialmente positivo).
- **Unidad de predicción:** una fila = `UGT × semana` (granularidad semanal).
  NO distrito (datos demasiado escasos: 9 de 18 distritos de interés tuvieron
  alguna vez una protesta) NI departamento completo (label se vuelve trivial
  por protestas urbanas de Chimbote/Huaraz no relacionadas con la operación).
- **UGTs de ANTAMINA (4):** Mina San Marcos, Huallanca, Valle Fortaleza,
  Huarmey. Cada una agrupa varios distritos (ver `src/scoring/ancash_datos.py`,
  constante `JERARQUIA`, 18 distritos en total).
- **Horizontes:** evaluar multi-horizonte `y_14`, `y_30`, `y_60` días. El
  modelo desplegado usa `y_30`.
- **Tipo de conflicto:** protestas y violencia que afecten directa o
  indirectamente a la operación de ANTAMINA en sus UGTs.
- **Label y features autoregresivas vienen exclusivamente de eventos dentro
  de los 18 distritos de las 4 UGTs** (BD de incidentes + formulario de
  monitoreo Antamina). Tres features de contexto (`tasa_pobreza`,
  `def_escalamiento_ancash`, `oefa_denuncias_mineria_12m`) son departamentales
  o de actividad genérica, usadas como señal ambiente, no como fuente del
  label.
- **Hueco de cobertura conocido:** Paramonga (distrito de Valle Fortaleza,
  único en provincia "Barranca - Lima") no tiene eventos en ninguna de las
  fuentes actuales — quedó sin cobertura de incidentes tras el cambio de
  fuente de datos (2026-06-30). Valle Fortaleza predice con los otros 9
  distritos.

## Fuentes de datos (todas filtradas o acotadas a Áncash)

- **BD de incidentes interna** (`src/data/dataIncidentes.xlsx`): eventos
  geolocalizados con distrito y fecha. Fuente principal del label y de las
  features autoregresivas (resolución semanal).
- **Formulario de monitoreo Antamina** (`src/data/dataFormulario.xlsx`): log
  de noticias específico de Antamina (~8,400 filas), categoría "Protestas,
  Paros y Bloqueos". Se suma a la BD de incidentes (deduplicado a 1 evento
  por distrito×fecha) — aporta ~3x más densidad de eventos de protesta dentro
  de las 4 UGTs que la BD de incidentes sola. Ver `src/scoring/ancash_datos.py`,
  `_cargar_formulario_protestas_raw()`.
- **Reportes situacionales de Antamina** (`src/data/2025/`, `src/data/2026/`,
  ~53 reportes Word/PowerPoint semanales/mensuales; `src/data/loader_reportes.py`):
  texto narrativo del equipo de relaciones de Antamina. ÚNICA fuente del
  proyecto que distingue explícitamente menciones A Antamina (no solo
  conflicto genérico de la zona) y que registra compromiso de las partes
  (mesas de diálogo, acuerdos, cronogramas) — antes marcado "PENDIENTE DE
  FUENTE" en el dashboard. Features: `rep_antamina_neg_4w` (tensión dirigida
  a Antamina), `rep_compromiso_4w` (mesas de diálogo/acuerdos activos).
- **Defensoría del Pueblo** (`src/data/scraper_defensoria.py`): reporte
  mensual de escalamiento de conflicto, acotado a Áncash. Feature de
  contexto departamental (rezago de 2 meses, anti-fuga).
- **Calendario** (feriados, elecciones, fechas críticas): features
  estructurales, no específicas de zona.
- **INEI** (`src/data/loader_inei.py`): tasa de pobreza de Áncash. Feature
  lenta (anual), acotada a Áncash.
- **OEFA/SINADA** (`src/data/loader_oefa.py`): denuncias ambientales públicas
  (portal de datos abiertos), filtradas a actividad minera dentro de las 4
  UGTs. Feature de **contexto acumulado** (ventana móvil de 12 meses),
  NO un predictor de corto plazo validado — se intentó comprobar si las
  denuncias anteceden a las protestas reales y el resultado fue inconcluso
  por falta de superposición temporal entre fuentes (el archivo de OEFA
  tiene ~12+ meses de rezago de publicación). Ver memoria del proyecto.
- **Pendientes de integrar** (mejoras identificadas, no implementadas):
  histórico de alertas propias, precio del cobre (commodities, zona minera),
  intensidad mediática (GDELT).

## Regla crítica: anti-fuga temporal (leakage)

- Toda feature de la semana `t` usa SOLO datos disponibles hasta el fin de la
  semana `t`. Nada posterior.
- El label `y_h` mira la ventana futura `[t+1, t+h]`. Features miran atrás.
- Defensoría se publica con rezago (reporte del mes M sale en M+1): usar
  SIEMPRE el último reporte efectivamente publicado a la fecha de corte.
- INEI: valor "as-of" (último publicado a la fecha, clip a 2025).

## Stack

- Python + venv. pandas, scikit-learn, joblib.
- Trabajo 100% LOCAL. Fuentes y datasets intermedios en `data/` (gitignored).
  Sin Neon, sin Postgres, sin Railway — no hay despliegue en la nube.

## Metodología (no negociable)

- **Validación temporal** (walk-forward / TimeSeriesSplit), NUNCA k-fold
  aleatorio.
- Métricas: **PR-AUC** y **recall a precisión fija**, NO accuracy (clases
  desbalanceadas).
- **Punto de control go/no-go:** el modelo debe superar un baseline trivial
  (tasa histórica de la UGT) por al menos +0.05 de PR-AUC. Si no, detener.

## Estructura de repositorio

```
ML alertas/
├── requirements.txt
├── CLAUDE.md
├── README_ANCASH.md       # documentación técnica detallada del modelo
├── notebooks/             # exploración
├── src/
│   ├── data/               # loaders: incidentes, calendario, INEI, Defensoría
│   ├── dataset/
│   │   └── build_ancash.py        # tabla maestra UGT × semana
│   ├── models/
│   │   └── train_ancash.py        # entrenamiento + punto de control go/no-go
│   └── scoring/
│       ├── ancash_datos.py            # índice de actividad observada (no es predicción)
│       ├── score_ancash.py            # scoring semanal del modelo
│       ├── dashboard_ancash.py        # dashboard Plotly (interno)
│       ├── dashboard_ancash_predictivo.py  # dashboard con plantilla de diseño + datos reales
│       └── assets/ancash_dashboard/    # plantilla visual, runtime, logo
├── models/                # artefactos entrenados (.pkl, gitignored)
└── data/                  # datasets intermedios (gitignored)
```

## Convenciones

- Código y comentarios pueden ir en español.
- Las fechas de corte y ventanas SIEMPRE explícitas en el código (evitar fuga).
- Antes de proponer un cambio al modelo, confirmar que el baseline existe y
  está medido.

## Estado actual (junio 2026)

### Modelo activo (Track A — Áncash, UGT × semana)
- Archivo: `models/modelo_v1_track_A_ancash.pkl`
- Algoritmo: regresión logística (`class_weight="balanced"`), horizonte `y_30`
- PR-AUC modelo (logistic_regression, el que se guarda): **0.8004** — ahora
  el mejor de los dos candidatos evaluados (random_forest: 0.792), gracias a
  las features de reportes Antamina (`rep_antamina_neg_4w`,
  `rep_compromiso_4w`). PR-AUC baseline (tasa histórica): 0.554 → GO (+0.246)
- Dataset: 444 observaciones (4 UGTs × ~111 semanas), 195 con protesta (43.9%,
  subió desde 28.6% al sumar `dataFormulario.xlsx` — ver memoria del proyecto
  `project_nuevas_fuentes_2026_06`)
- **Umbral de alerta calibrado** (`src/models/recalibrar_umbral.py`,
  2026-06-30, sobre predicciones out-of-fold walk-forward): ALTO ≥ 80%
  (precisión 75%, recall 69%, tasa de alerta 59% con el modelo actual),
  MEDIO 44–79% (tasa histórica de protesta como referencia de "más riesgo
  que el promedio"), BAJO < 44%. Reemplaza el 55%/35% que era solo
  referencia visual.

### Operación semanal
```bash
python -m src.scoring.score_ancash               # predicción por UGT (consola)
python -m src.scoring.dashboard_ancash_predictivo # dashboard interactivo
```

### Reconstruir el pipeline desde cero
```bash
python -m src.data.loader_calendario
python -m src.data.loader_inei
python -m src.data.scraper_defensoria   # requiere conexión, opcional si ya existe el CSV
python -m src.data.loader_oefa          # requiere conexión, opcional si ya existe el parquet
python -m src.data.loader_reportes      # parsea src/data/2025/ y 2026/ (Word/PowerPoint)
python -m src.dataset.build_ancash
python -m src.models.train_ancash
python -m src.models.recalibrar_umbral  # recalibrar el umbral si el modelo cambió
```

### Fase de validación operativa
Cada viernes anotar: ¿hubo conflicto/protesta real esta semana en cada UGT?
(sí/no). Con varias semanas de datos se puede calcular el recall operativo
real y contrastarlo contra el PR-AUC de validación.

### Limitaciones honestas (ver README_ANCASH.md, sección 9)
Dataset todavía pequeño (444 filas, solo 71 eventos reales sostienen el
label), sin umbral formal, features de contexto departamental limitadas.
