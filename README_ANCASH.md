# Modelo Predictivo de Conflictos Sociales — Región Áncash (ANTAMINA)

Documentación técnica del subsistema **Track A regional de Áncash**, pensada
para que alguien con conocimientos básicos/intermedios de ingeniería de
sistemas pueda entender de punta a punta cómo se construyó, por qué se tomó
cada decisión, y cómo reproducirlo o extenderlo.

Este documento cubre **solo Áncash**. El resto del proyecto (Track B y las
otras zonas de Track A) sigue la misma filosofía pero con datasets y modelos
separados — ver `CLAUDE.md` para el contexto general.

---

## 1. El problema que resuelve

Antamina necesita saber, **con anticipación**, qué tan probable es que ocurra
una protesta nueva en cada una de sus zonas operativas durante el próximo mes.
No es lo mismo que "monitorear lo que ya está pasando" (eso ya existe en el
dashboard de actividad observada): el objetivo aquí es **pronosticar**.

Esto se modela como una **clasificación binaria supervisada con componente
temporal**: para una unidad espacial y una semana dadas, predecir si habrá o
no una protesta nueva en una ventana futura.

---

## 2. Unidad de análisis: ¿por qué UGT y no distrito ni departamento?

Se evaluaron tres niveles de granularidad espacial antes de elegir:

| Nivel | Problema |
|---|---|
| **Departamento** (Áncash completo) | El label se vuelve casi trivial: hay protestas urbanas en Chimbote/Huaraz que no tienen nada que ver con la operación minera de Antamina, pero "contaminan" la etiqueta departamental subiéndola artificialmente. |
| **Distrito** (18 distritos de interés) | Demasiado disperso: de los 18 distritos, solo 9 tuvieron alguna vez un evento de protesta registrado. Con tan pocos ejemplos por distrito, ningún modelo puede aprender un patrón confiable — es ruido, no señal. |
| **UGT** (Unidad de Gestión Territorial) ✅ | Es la agrupación operativa real que usa Antamina (cada UGT agrupa varios distritos cercanos a una operación). Da suficiente densidad de eventos por unidad y es directamente accionable para el cliente. |

Las 4 UGTs y su jerarquía territorial (`src/scoring/ancash_datos.py`, `JERARQUIA`):

| UGT | Provincias | Distritos |
|---|---|---|
| Mina San Marcos | Huari | San Marcos, Chavín de Huántar, Huachis, San Pedro de Chana |
| Huallanca | Bolognesi | Huallanca, Aquía, Chiquián |
| Valle Fortaleza | Bolognesi, Recuay, Barranca-Lima | Cajacay, Antonio Raimondi, Colquioc, Huayllacayán, Catac, Pampas Chico, Marca, Llacllín, Pararín, Paramonga |
| Huarmey | Huarmey | Huarmey |

Nótese que **UGT no respeta límites provinciales estrictos** (Valle Fortaleza
cruza tres provincias) — es deliberado, porque la unidad de interés es la
operación, no la división político-administrativa.

---

## 3. Las dos capas del sistema (no confundir)

El dashboard de Áncash mezcla dos cosas que **deben mantenerse conceptualmente
separadas**:

1. **Índice de actividad observada** (`src/scoring/ancash_datos.py`) — describe
   el pasado/presente: "¿cuánta conflictividad reciente hay en este distrito?".
   No predice nada, es una fotografía ponderada del historial reciente.
2. **Modelo predictivo Track A** (`src/dataset/build_ancash.py` →
   `src/models/train_ancash.py` → `src/scoring/score_ancash.py`) — sí predice:
   "¿qué probabilidad hay de una protesta nueva en los próximos 30 días?".

Esta sección cubre primero la capa 1 (más simple) y luego la capa 2 (el
modelo real).

---

## 4. Capa 1 — Índice de actividad observada

Archivo: `src/scoring/ancash_datos.py`. No usa machine learning, es una
fórmula determinística.

### 4.1 Fuente

`src/data/BD_Incidentes - copia.xlsx` (hoja `INCIDENTES`), base interna con
eventos geolocalizados: fecha, departamento, distrito, motivo, título,
lat/lon. Se filtra a `DEPTOS_VALIDOS = {"ANCASH", "LIMA"}` y se normaliza
texto (`_norm`: minúsculas + sin tildes) para machear nombres de distrito de
forma robusta.

### 4.2 Categorización del motivo

Cada fila tiene un `MOTIVO` libre (texto). Se mapea a una de 4 categorías via
el diccionario `MOTIVO_A_CATEGORIA`:

- **PROTESTA**: bloqueo, reclamo, reunión, evento social, incidente fundo/mina, etc.
- **POLITICA**: movimiento político, político-electoral.
- **VIOLENCIA**: homicidio, sicariato, secuestro, extorsión, robo, etc.
- **OTRO**: cualquier motivo no mapeado (se descarta del índice).

### 4.3 Fórmula del índice (decaimiento exponencial + normalización)

Para cada evento, el peso es:

```
peso_evento = peso_categoría × exp(-días_desde_el_evento / 180)
```

- `peso_categoría` (constante `PESOS`): PROTESTA=1.0, POLITICA=0.8, VIOLENCIA=0.45
  (la conflictividad social pesa más que la inseguridad común para el
  propósito de alerta de *conflictos*, no de seguridad ciudadana en general).
- `180` días (`TAU_DIAS`) es la "vida media" del decaimiento: un evento de
  hace 180 días pesa `e^-1 ≈ 37%` de lo que pesaría si fuera hoy. Esto hace
  que el índice priorice lo reciente sin descartar brutalmente lo de hace
  unos meses.

El score crudo de un distrito es la suma de los pesos de todos sus eventos
históricos. Luego se normaliza a una escala 0-100:

```
score = 100 × √(min(score_crudo / percentil_95, 1))
```

- Se usa el **percentil 95** (no el máximo) como referencia para que un solo
  distrito con actividad extrema no aplaste la escala de todos los demás.
- La **raíz cuadrada** comprime el rango: sin ella, la mayoría de distritos
  con actividad baja-media quedarían apilados cerca de 0 y el rango medio
  (donde más importa distinguir "medio" de "alto") perdería resolución.

### 4.4 Estados y acción recomendada

```
score >= 70 → Alto (rojo)    → Mesa de diálogo urgente
score >= 55 → Alto (naranja) → Intervención preventiva
score >= 35 → Medio (ámbar)  → Monitoreo cercano
score <  35 → Bajo (verde)   → Vigilancia rutinaria
```

Esto es **descriptivo**, no es la salida del modelo de ML — son umbrales de
negocio aplicados sobre una métrica observacional.

---

## 5. Capa 2 — El modelo predictivo

### 5.1 Variable objetivo (target)

`y_30`: binaria, **"¿hubo al menos una protesta nueva en esta UGT durante los
30 días siguientes a esta semana?"** (también existen `y_14` y `y_60` en el
dataset, calculados igual pero con otra ventana; el modelo desplegado usa
`y_30`).

Importante: el label sale de la **misma BD de incidentes**, filtrada solo a
categoría `PROTESTA` (no VIOLENCIA — la violencia es una *feature*, no el
fenómeno que se quiere predecir).

### 5.2 Construcción anti-fuga del label

Código clave (`src/dataset/build_ancash.py`, función `_labels`):

```python
ini = sv + np.timedelta64(1, "D")     # empieza el día después de la semana
fin = sv + np.timedelta64(h, "D")     # termina h días después
hay = (searchsorted(eventos, ini) < searchsorted(eventos, fin))
hay[fin > fecha_corte] = NaN          # si la ventana se sale del dato disponible, no se etiqueta
```

Dos reglas de anti-fuga (leakage) se cumplen aquí:
1. El label **mira estrictamente hacia adelante** (`ini` es el día siguiente
   a la semana, nunca incluye el presente).
2. Si la ventana de 30 días se extiende más allá de `FECHA_CORTE` (el último
   punto con datos confiables), esa fila se marca `NaN` y se descarta — de lo
   contrario se estaría infiriendo "no hubo protesta" de la *ausencia de
   datos futuros*, que es distinto de "no hubo protesta realmente".

### 5.3 Features (16 variables)

| Grupo | Variable | Qué mide |
|---|---|---|
| Incidentes (autoregresivas) | `inc_prot_1w` | protestas en la última semana |
| | `inc_prot_4w` | protestas en las últimas 4 semanas (suma móvil) |
| | `inc_viol_1w`, `inc_viol_4w` | igual pero violencia |
| | `delta_prot` | aceleración: semana actual vs. promedio de las 4 previas |
| | `racha_prot` | semanas consecutivas con al menos 1 protesta |
| | `dias_desde_ultima_prot` | qué tan "fresca" es la última protesta conocida |
| Calendario | `n_feriados`, `es_semana_electoral`, `dias_hasta_eleccion`, `es_fecha_critica`, `mes`, `trimestre`, `semana_iso` | estructura temporal/política compartida con el resto del proyecto |
| Contexto departamental | `tasa_pobreza` | INEI, valor anual "as-of" (solo Áncash) |
| | `def_escalamiento_ancash` | Defensoría del Pueblo, departamental, con **2 meses de rezago** |

Todas estas features se calculan **por UGT** salvo las dos últimas, que son
departamentales y se replican igual para las 4 UGTs (proxy: si no hay dato a
nivel UGT, se usa el dato del nivel territorial inmediatamente superior — el
mismo patrón que usa el resto del proyecto con `ZONA_A_DEPTO_PADRE`).

El rezago de 2 meses en `def_escalamiento_ancash` no es arbitrario: la
Defensoría publica su reporte del mes M en M+1, así que al construir la
feature de una semana dada solo se usa el último reporte que *ya estaría
publicado* en esa fecha (nunca el del mes en curso).

### 5.4 Esqueleto del dataset

```python
UGTS = ["Mina San Marcos", "Huallanca", "Valle Fortaleza", "Huarmey"]
FECHA_INICIO = "2024-01-01"
FECHA_CORTE  = "2026-04-13"
```

Se genera el producto cartesiano UGT × semana (`itertools.product`) para
tener una fila por cada combinación, **incluyendo semanas sin ningún
evento** (mayoría de los casos) — esto es necesario porque el modelo necesita
ver tanto los casos positivos como los negativos para aprender a discriminar.

Resultado: **444 filas** (4 UGTs × ~111 semanas), guardado en
`data/processed/master_ancash_ugt.parquet`.

Balance del label `y_30`: 28.6% positivos (127/444), pero desigual por UGT —
Huarmey 47%, Mina San Marcos 33%, Huallanca 18%, Valle Fortaleza 16%. Este
desbalance moderado (no extremo, no trivial) es justamente lo que se buscaba
verificar *antes* de construir el pipeline completo, para no repetir el
problema de "label trivial" que bloqueó este Track en departamento completo.

### 5.5 Algoritmo y pipeline de entrenamiento

Archivo: `src/models/train_ancash.py`. Se probaron dos modelos:

```python
"logistic_regression": Pipeline([
    SimpleImputer(strategy="median", keep_empty_features=True),
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
]),
"random_forest": Pipeline([
    SimpleImputer(strategy="median", keep_empty_features=True),
    RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced"),
]),
```

- **`SimpleImputer(keep_empty_features=True)`**: rellena NaN con la mediana.
  El flag `keep_empty_features` evita que columnas que en algún fold de
  entrenamiento no tuvieron *ningún* valor observado (ej. `dias_hasta_eleccion`
  en ciertos rangos) sean eliminadas silenciosamente — error real que se dio
  durante el desarrollo.
- **`class_weight="balanced"`**: como el 28.6% de positivos no es 50/50, el
  algoritmo penaliza más los errores sobre la clase minoritaria (protesta),
  en vez de simplemente "aprender a decir siempre que no" para maximizar
  exactitud bruta.
- **`max_depth=4`** en Random Forest: con solo 444 filas, un árbol profundo
  memorizaría ruido en vez de aprender un patrón generalizable (overfitting).

**Modelo elegido: `logistic_regression`.** Random Forest tuvo un PR-AUC casi
idéntico (0.593 vs 0.589) pero con una desviación estándar mucho mayor entre
folds (±0.21 vs ±0.09) — con un dataset tan chico, la estabilidad pesa más que
una décima de PR-AUC adicional.

### 5.6 Validación: walk-forward, nunca k-fold aleatorio

```python
TimeSeriesSplit(n_splits=4)
```

Esto es una regla de oro del proyecto entero (ver `CLAUDE.md`). La razón:
con datos de serie temporal, un k-fold aleatorio normal mezclaría semanas del
"futuro" en el conjunto de entrenamiento al predecir el "pasado" — eso
sobreestima artificialmente qué tan bueno es el modelo, porque en producción
real nunca tendrías datos futuros disponibles. `TimeSeriesSplit` garantiza
que cada fold de prueba sea *posterior* a su fold de entrenamiento
correspondiente, simulando honestamente cómo se usaría el modelo semana a
semana en producción.

De los 4 splits configurados, solo 3-4 resultan "válidos" en cada corrida —
un fold se descarta si su porción de entrenamiento o de prueba termina con
una sola clase presente (ej. ningún positivo), porque ahí ninguna métrica de
clasificación tiene sentido.

### 5.7 Métricas: PR-AUC y recall a precisión fija (nunca accuracy)

- **Por qué no accuracy**: con 71.4% de negativos, un modelo que *siempre*
  dice "no" ya tendría 71.4% de exactitud sin haber aprendido nada útil.
  Accuracy es engañosa en clases desbalanceadas.
- **PR-AUC** (Precision-Recall Area Under Curve): resume qué tan bien el
  modelo distingue la clase positiva a través de todos los umbrales
  posibles, sin la distorsión que el desbalance le genera al más conocido
  ROC-AUC.
- **Recall a precisión fija** (`recall@P50` en el código): de todas las
  protestas reales, qué porcentaje el modelo efectivamente captura, *cuando
  se exige que al menos el 50% de sus alertas sean ciertas* (precisión
  mínima de 50%). Es una métrica más cercana a "¿esto es operacionalmente
  útil?" que el PR-AUC solo.

### 5.8 Punto de control go/no-go

Antes de confiar en cualquier modelo, se lo compara contra dos baselines
triviales bajo el mismo esquema de validación walk-forward:

- `siempre_negativo`: predecir 0% de probabilidad siempre.
- `tasa_historica`: predecir la tasa de positivos observada en el
  entrenamiento (sin mirar ninguna feature).

```
Mejor baseline:  tasa_historica = 0.3608
Mejor modelo:    logistic_regression = 0.5931
Diferencia:      +0.2323   (umbral de GO: +0.05)
VEREDICTO:       GO
```

El modelo solo se guarda si supera el baseline por al menos `UMBRAL_GO=0.05`
de PR-AUC. Esta vez lo superó por más de 4 veces ese margen.

### 5.9 Modelo final guardado

```python
joblib.dump({
    "model": modelo_final,            # entrenado sobre las 444 filas completas
    "feature_cols": FEATURES,         # las 16 variables, en el orden exacto que espera el modelo
    "track": "A", "region": "Áncash", "unidad": "UGT × semana",
    "target": "y_30",
    "pr_auc_cv": 0.5931, "pr_auc_baseline": 0.3608,
    "modelo_tipo": "logistic_regression",
    "trained_on": "<fecha>",
}, "models/modelo_v1_track_A_ancash.pkl")
```

Nótese que el modelo final se reentrena sobre **todos** los datos (no solo
sobre un fold) una vez que pasó el control de calidad — los folds de
`TimeSeriesSplit` sirven para *medir* qué tan bueno sería el modelo en
producción, no para producir el modelo que efectivamente se despliega.

---

## 6. Scoring semanal (poner el modelo a trabajar)

Archivo: `src/scoring/score_ancash.py`. Reutiliza literalmente las mismas
funciones de construcción de features que `build_ancash.py` (no se duplica
lógica) para garantizar que las features de "hoy" se calculen exactamente
igual que las que vio el modelo en entrenamiento.

```python
def predecir(lunes_actual=None):
    paquete = joblib.load(MODELO_PATH)
    df = calcular_features_actuales(lunes_actual)   # mismas funciones que el dataset de entrenamiento
    X = df[paquete["feature_cols"]]
    df["probabilidad"] = paquete["model"].predict_proba(X)[:, 1]
    ...
```

Salida típica (semana del 2026-06-15):

| UGT | Probabilidad | Nivel |
|---|---|---|
| Mina San Marcos | 65.1% | ALTO |
| Valle Fortaleza | 62.2% | ALTO |
| Huarmey | 61.2% | ALTO |
| Huallanca | 34.9% | BAJO |

Los umbrales ALTO/MEDIO/BAJO mostrados aquí (55% / 35%) son **de visualización**,
no un umbral estadísticamente calibrado como el que sí existe para el modelo
de Track B (`src/models/recalibrar_umbral.py`). Calibrar uno equivalente para
Áncash sigue pendiente.

---

## 7. Dashboard (`src/scoring/dashboard_ancash.py`)

Genera un único archivo HTML autocontenido (`data/processed/dashboard_ancash.html`)
con Plotly embebido vía CDN. Combina:

- La sección de **predicción** (`_obtener_prediccion`, `_tarjetas_prediccion`):
  llama a `score_ancash.predecir()` y muestra las 4 tarjetas con su badge
  `PR-AUC` — claramente etiquetada como modelo, distinta del resto.
- La capa de **actividad observada** (mapa de distritos, serie mensual,
  donut de composición) que viene de `ancash_datos.py`.
- Filtros jerárquicos UGT → Provincia → Distrito implementados en JavaScript
  puro, con los datos de los 18 distritos embebidos como JSON en el HTML
  (no requiere backend ni servidor).
- Secciones explícitamente marcadas `PENDIENTE DE FUENTE` para lo que el
  proyecto aún no tiene (drivers socioeconómicos detallados, compromiso de
  las partes) — para no insinuar que existe un dato que en realidad no está
  disponible.

---

## 8. Cómo reproducir todo desde cero

Requiere que ya existan los insumos compartidos con el resto del proyecto
(se generan una sola vez, no son específicos de Áncash):

```bash
python -m src.data.loader_calendario     # -> data/interim/calendario.parquet
python -m src.data.loader_inei           # -> data/interim/inei_pobreza.parquet
# data/raw/defensoria/defensoria_historico.csv ya debe existir (scraper_defensoria.py)
```

Luego, el pipeline propio de Áncash, en orden:

```bash
python -m src.dataset.build_ancash       # construye master_ancash_ugt.parquet (444 filas)
python -m src.models.train_ancash        # entrena, valida go/no-go, guarda el .pkl si pasa
python -m src.scoring.score_ancash       # imprime la predicción de la semana actual
python -m src.scoring.dashboard_ancash   # genera y abre el dashboard HTML
```

---

## 9. Limitaciones honestas (léase antes de presentar el modelo a alguien)

- **Dataset pequeño**: 444 filas, 127 positivos, solo 3-4 folds de validación
  válidos. El GO es real, pero con mucho menos margen de certeza estadística
  que el modelo de Track B (que tiene ~3 años de historia semanal).
- **Solo 71 eventos reales** sostienen todo el label histórico. Cualquier
  cambio en cómo se registra la BD de incidentes (más o menos reportes)
  afecta directamente la calidad del modelo.
- **`tasa_pobreza` es anual y casi constante** dentro del rango de datos —
  aporta poca señal real pese a estar en la lista de features.
- **`def_escalamiento_ancash` es departamental**, no distingue entre las 4
  UGTs — es la misma feature aplicada a las 4 filas de cada semana.
- **Sin umbral de alerta calibrado formalmente** (a diferencia de Track B) —
  hoy el dashboard usa 55%/35% solo como referencia visual.
- **No incluye aún** `alertas_propias` (el histórico interno de 3 años que
  sí alimenta Track B), ni commodities (precio del cobre, relevante para una
  zona minera), ni GDELT — todas son mejoras identificadas pero no
  implementadas todavía.

---

## 10. Glosario rápido (para quien recién empieza)

| Término | Significado en este proyecto |
|---|---|
| **Feature** | Un dato de entrada que el modelo usa para razonar (ej. cuántas protestas hubo la semana pasada) |
| **Label / target** | La respuesta correcta que se le enseña al modelo durante el entrenamiento (`y_30`) |
| **Leakage (fuga de datos)** | Cuando, por error, el modelo "ve" información del futuro durante el entrenamiento, lo que infla artificialmente sus métricas |
| **Walk-forward / TimeSeriesSplit** | Forma de validar un modelo de serie temporal que respeta el orden cronológico, nunca mezclando pasado y futuro |
| **PR-AUC** | Métrica que resume qué tan bien el modelo distingue la clase positiva, robusta a clases desbalanceadas |
| **Baseline trivial** | Un predictor "tonto" (ej. siempre la tasa histórica) contra el que se debe comparar cualquier modelo antes de confiar en él |
| **class_weight="balanced"** | Ajuste que evita que el modelo ignore la clase minoritaria solo porque es menos frecuente |
