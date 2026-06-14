# Model Card — Dupin `m-v1`

Tarjeta del modelo de scoring de fraude transaccional de Dupin. Sigue el espíritu
de las *model cards* de Mitchell et al. (2019): qué hace el modelo, cómo se
evaluó, y —sobre todo— qué **no** se debe asumir de él.

| | |
|---|---|
| **Modelo** | `m-v1` · LightGBM (gradient boosting tabular) |
| **Features** | `feat-v1` · 13 features de comportamiento, causales |
| **Tarea** | Clasificación binaria por transacción: P(fraude) |
| **Dataset** | PaySim (sintético), CC BY-SA 4.0 |
| **Artefacto** | `gs://dupin-dupin-artifacts/m-v1/` (model + schema + umbrales + metadata) |
| **Serving** | Cloud Run · `POST /v1/score` · escala a cero |

---

## Uso previsto

- **Para qué sirve:** demostrar la *metodología correcta* de construir, evaluar y
  desplegar un sistema antifraude — features causales sin fuga, evaluación
  temporal honesta, costo de negocio, serving con explicación y monitoreo.
- **Para qué NO sirve:** **no es un modelo de producción.** Está entrenado sobre
  datos sintéticos; las métricas absolutas no se transfieren a fraude real. No
  debe usarse para decisiones financieras reales.
- **Usuarios:** revisión técnica, portafolio, enseñanza de evaluación honesta.

## Datos

- **Fuente:** PaySim — Lopez-Rojas, Elmir & Axelsson (2016), EMSS. CC BY-SA 4.0,
  vía Kaggle `ealaxi/paysim1`. 6.362.620 transacciones, 30 días (`step` = 1 hora).
- **Prevalencia de fraude:** 0.129% global; 0.30% en la superficie de scoring.
- **Superficie:** el fraude vive **solo** en `TRANSFER` y `CASH_OUT`; el modelo
  puntúa solo esos tipos (2.77M filas). El resto se auto-aprueba.
- **Fuga de etiqueta evitada:** las columnas `oldbalanceOrg`, `newbalanceOrig`,
  `oldbalanceDest`, `newbalanceDest` codifican el etiquetado (PaySim anula las
  transacciones de fraude) → **prohibidas como features**. Verificado por
  auditoría (un modelo sobre ellas + split aleatorio alcanza ROC-AUC ~0.99: fuga).

## Features (`feat-v1`)

Todas causales: se computan solo con información **anterior** al instante de la
transacción, acumuladas sobre la cuenta **receptora** (`nameDest`) —los
originadores son de un solo uso (99.85% únicos) y no tienen historia—.

`log_amount`, `type_transfer`, `hour`, `dest_prior_count`, `dest_is_new`,
`dest_amount_ratio`, `dest_amount_z`, `dest_recency`, `dest_cnt_24h`,
`dest_cnt_168h`, `dest_amt_sum_24h`, `dest_amt_sum_168h`, `orig_prior_count`.

Top de importancia: `log_amount`, `dest_amt_sum_168h`, `dest_amount_ratio`,
`dest_recency` — el comportamiento del receptor domina. `orig_prior_count` es
peso muerto (confirma el single-use).

## Evaluación

**Split temporal estricto** (la vara de medir honesta): train `[0, 432)`, val
`[432, 480)`, test `[480, 744)`. Sin solapamiento; ninguna información del futuro
entra al entrenamiento. El split aleatorio se reporta solo como diagnóstico de
optimismo.

Punto de operación de cabecera: **recall @ revisar ≤1% de operaciones**.

### Resultados (test temporal)

| Punto de operación | Recall | Precision | Review rate |
|---|---|---|---|
| Desplegado (umbral val→test) | 21.0% | 95.4% | 0.47% |
| **Envolvente al 1% (capacidad)** | **35.3%** | 75.4% | 1.00% |
| Auto-block (tier) | 1.74% | 100% | 0.04% |

- PR-AUC temporal: **0.594** (prevalencia 2.14% → ~28× sobre el azar).
- Envolvente: ~54% recall al 2%, ~78% al 5%.

### El gap honesto (resultado central)

| Configuración | Recall@1% | Lectura |
|---|---|---|
| Balance crudo + split aleatorio | **99.8%** | fantasía: fuga de etiqueta + temporal |
| Honesto + split temporal | **35.3%** | el número desplegable |

El gap se descompone en dos ejes: **fuga de etiqueta** (columnas de balance) y
**fuga temporal** (aleatorio vs temporal). Reportar el 99.8% como resultado sería
deshonesto; el 35.3% es el número real.

## Umbrales y decisión

Bundle `m-v1`: `review = 0.9984`, `block = 0.9999` (top 1% / 0.1% sobre
validación). Decisión: `score ≥ block → block`; `≥ review → review`; si no
`approve`. Los umbrales son artefactos versionados, no constantes en código.

**Transferencia de umbral:** el umbral fijado en validación (prevalencia 3.96%)
queda conservador para el test (2.14%), gastando 0.47% del presupuesto de 1% — por
eso el punto desplegado (21%) cae bajo la envolvente (35.3%). En producción se
recupera recalibrando el umbral sobre datos recientes (monitoreo de drift).

## Explicabilidad

Cada predicción devuelve la razón del flag: atribución nativa de LightGBM
(`pred_contrib`) traducida a lenguaje legible ("monto 12× el promedio del
receptor", "3ra transacción al receptor en 24h"). La razón es parte del contrato
de respuesta.

## Limitaciones

- **Dato sintético.** Los patrones de PaySim son más simples y separables que el
  fraude real; las métricas absolutas **no** se transfieren a producción. El valor
  es la metodología, no el número.
- **Sin grafos de entidades.** El fraude organizado se detecta mejor con
  relaciones entre entidades; v1 es tabular por entidad individual.
- **Scoring por transacción, no por sesión.** El fraude de secuencia/sesión queda
  fuera de alcance.
- **Umbral único global.** Sin segmentación por cliente/canal.
- **Estado cold-start en serving.** El estado de entidades arranca vacío y se
  calienta con el tráfico; en producción iría respaldado por un feature store.

## Ética y riesgos

- Datos sintéticos, sin PII real.
- No emite consejo financiero ni decisiones reales.
- El sesgo del simulador (no del mundo real) limita cualquier conclusión sobre
  poblaciones reales.

## Reproducibilidad

Semillas fijas (42), dependencias pineadas (`pyproject.toml`), bundle versionado,
features acopladas a `feature_version`. Pipeline: notebooks `01`–`05` + suite de
tests (`pytest`, 43/43).
