# Dupin — Real-Time Transaction Fraud Scoring

Sistema de scoring de riesgo de fraude transaccional en tiempo real para fintech. Dada una transacción individual dentro de un flujo de operaciones, el sistema emite un score de riesgo continuo, una decisión (`approve` / `review` / `block`) y la razón explícita del flag, con latencia medida bajo carga.

**Estado:** prototipo v0.1 · modelo `m-v1` · features `feat-v1`
**Stack:** GCP (GCS + Colab + Cloud Run) · gradient boosting tabular · FastAPI · dashboard de monitoreo en vivo

---

## TL;DR

Dupin es un sistema antifraude de extremo a extremo: ingesta de transacciones → ingeniería de features de comportamiento → evaluación bajo régimen temporal honesto → serving en Cloud Run → dashboard de monitoreo. El clasificador es un modelo de gradient boosting (no una red neuronal: en datos tabulares desbalanceados los árboles con boosting ganan en rendimiento, velocidad e interpretabilidad).

**El eje del proyecto no es el clasificador.** Un clasificador de fraude decente es casi commodity. El eje es el **régimen de evaluación temporal**: la mayoría de las demostraciones de fraude evalúan con un split aleatorio que filtra información del futuro al pasado y produce métricas fantasiosas. Dupin evalúa con split temporal estricto —entrena en el pasado, prueba en el futuro, como ocurre en producción— y reporta la caída honesta frente al número optimista del split aleatorio. Ese contraste es el resultado, no el AUC.

**Honestidad de origen:** el sistema se construye sobre datos sintéticos públicos. No pretende ser un modelo de producción; demuestra la metodología correcta de construir, evaluar y desplegar uno. Las limitaciones del dato sintético se declaran explícitamente en su sección.

---

## Decisiones de ingeniería clave

- **Split temporal como invariante de evaluación.** Ninguna métrica obtenida con split aleatorio cuenta como resultado desplegable. El número de cabecera es siempre el que se obtiene entrenando sobre un periodo anterior y evaluando sobre uno posterior, sin solapamiento temporal. El split aleatorio se reporta solo como diagnóstico, para cuantificar cuánto optimismo introduce.
- **Las features son comportamiento, no transacción aislada.** El fraude vive en la desviación respecto al patrón histórico de la entidad (cuenta, tarjeta, comercio), no en los campos crudos de una transacción. La capa de features es el núcleo intelectual del sistema.
- **Sin fuga temporal en las features.** Toda feature de agregación se computa usando únicamente información disponible *antes* del instante de la transacción evaluada. Una feature que use el futuro infla la métrica y rompe en producción. Esta es la falla silenciosa más común en proyectos de fraude.
- **Métricas de negocio, no de laboratorio.** El desbalance extremo hace que accuracy y ROC-AUC engañen. La cabecera es precision-recall y el costo de negocio a un umbral de operación: cuánto fraude se atrapa contra cuántos clientes legítimos se molestan.
- **El artefacto es un bundle versionado.** Un modelo de Dupin no es un archivo; es modelo + esquema de features + umbral de decisión + metadatos de entrenamiento, publicados y cargados juntos. El serving rechaza arrancar si falta un componente.

---

## Arquitectura del sistema

El sistema se divide en dos planos conectados por un artefacto de modelo versionado almacenado en GCS, replicando el patrón de separación offline/online.

### Plano offline (features + entrenamiento + evaluación)

Corre en Google Colab (CPU; el boosting no requiere GPU). Lee los datos crudos desde GCS, construye las features de comportamiento, entrena el modelo, lo evalúa bajo régimen temporal, selecciona el umbral de operación y publica el bundle completo a GCS.

### Plano online (serving + monitoreo)

Corre en Cloud Run. Al arrancar, descarga el bundle desde GCS (modelo + esquema de features + umbral). Expone `POST /v1/score` para puntuar transacciones y sirve el dashboard de monitoreo. Escala a cero cuando no recibe tráfico.

### Flujo completo de datos

```
[Dataset sintético público]
        │  (descarga máquina-a-máquina: fuente → GCS, sin pasar por local)
        ▼
[GCS: raw/]                    ← transacciones crudas, inmutables
        │
        ▼
[features/build_features.py]   ← única fuente de verdad transacción→vector
        │   Agregaciones temporales por entidad, sin fuga de futuro.
        │   Importado por el pipeline de entrenamiento Y por el serving.
        ▼
[GCS: features/feat-v1/]        ← matriz de features versionada
        │
        ▼
[training/train.py]
        │   Split temporal: train = periodo anterior, test = periodo posterior.
        │   Entrena gradient boosting. Maneja desbalance vía pesos/scale.
        ▼
[evaluation/evaluate.py]
        │   Métricas PR + curva de costo de negocio.
        │   Selecciona umbral de operación sobre validación temporal.
        │   Reporta también el split aleatorio como diagnóstico de optimismo.
        ▼
[GCS: artifacts/m-v1/]          ← bundle: model + feature_schema + threshold + metadata
        │
        ▼
[serving/model_runtime.py]
        │   Singleton cargado al arrancar Cloud Run.
        │   El mismo features/ del entrenamiento. Cero reimplementación.
        ▼
[serving/app.py]                ← FastAPI
        │   POST /v1/score: features → modelo → umbral → score + razón.
        │   Sirve el dashboard estático y el stream de eventos.
        ▼
[serving/dashboard/]            ← monitoreo en vivo del flujo de scoring
```

### Por qué `features/` es la invariante más crítica

El train/serve skew —que las features se computen distinto en entrenamiento que en producción— es la fuente más común de degradación silenciosa en sistemas de ML tabular. Si una agregación, una ventana temporal o un default de valor faltante difieren entre las dos etapas, el modelo recibe en producción vectores que nunca vio en entrenamiento.

En Dupin, `features/` es importado tanto por `training/` como por `serving/`. Es físicamente imposible que difieran.

---

## Estructura del repositorio

```
dupin/
├── data/
│   ├── download/
│   │   └── colab_ingest.ipynb      # descarga fuente→GCS sin pasar por local
│   ├── schema.py                   # esquema tipado de la transacción cruda
│   └── splits.py                   # define cortes temporales train/val/test
├── features/
│   ├── config.py                   # FeatureConfig: ventanas, entidades, defaults
│   ├── build_features.py           # transacción→vector: agregaciones sin fuga
│   ├── entity_state.py             # estado histórico por entidad (cuenta/tarjeta)
│   └── tests/                      # fuga temporal, paridad, defaults, dtype/shape
├── training/
│   ├── model.py                    # construcción y config del gradient boosting
│   ├── train.py                    # split temporal → fit → publish_bundle
│   └── export.py                   # publish_bundle: model + schema + threshold → GCS
├── evaluation/
│   ├── metrics.py                  # precision-recall, curva de costo, umbral
│   ├── evaluate.py                 # evalúa bundle bajo régimen temporal
│   ├── leakage_audit.py            # verifica ausencia de fuga de futuro
│   └── report.py                   # tabla de resultados, curvas, JSON de resumen
├── serving/
│   ├── app.py                      # FastAPI: /v1/score, stream, estáticos
│   ├── model_runtime.py            # ModelRuntime singleton: carga bundle, predict()
│   ├── explain.py                  # razón del flag por predicción (feature attribution)
│   ├── Dockerfile                  # multi-stage, slim, usuario no-root
│   ├── dashboard/                  # frontend de monitoreo en vivo
│   └── tests/                      # endpoints, merge, health sin modelo, campos
├── infra/
│   ├── 00_enable_apis.sh           # habilita Cloud Run, GCS, Cloud Build, AR
│   ├── 01_buckets.sh               # crea buckets raw, features, artifacts
│   ├── 02_artifact_registry.sh
│   ├── 03_service_accounts.sh      # SA de serving con solo objectViewer sobre artifacts
│   └── 04_deploy.sh                # Cloud Build + run deploy con verificación de bundle
├── docs/
│   ├── model_card.template.md
│   └── model_card_m-v1.md          # métricas reales + limitaciones del dato sintético
├── cloudbuild.yaml
├── config.example.yaml
└── pyproject.toml                  # dependencias pineadas
```

---

## Fase 0 — Decisión, datos y licencias

Fase de diseño, sin código. Define el alcance y previene el desperdicio aguas abajo.

- **Selección del dataset.** Elegir un dataset sintético público de transacciones con licencia permisiva y etiqueta de fraude. Candidatos: simuladores de pagos móviles tipo PaySim y datasets de fraude de tarjeta. Criterio: licencia clara para uso/publicación, etiqueta de fraude presente, marca temporal por transacción (indispensable para el split temporal) e identificador de entidad (cuenta/tarjeta) para las features de comportamiento.
- **Verificación de licencia.** Documentar la licencia exacta y clasificar el dataset como publicable. Sin marca temporal por transacción o sin identificador de entidad, el dataset no sirve para este proyecto: descartar antes de invertir.
- **Definición de la pregunta.** Fijar en una frase qué predice el sistema y a qué nivel (por transacción), y qué se contará como resultado (el gap temporal-vs-aleatorio). Esto se escribe antes de tocar datos.
- **Setup de GCP.** Proyecto nuevo bajo la misma cuenta Google, aislado para contabilidad de costos limpia. Presupuesto con alerta configurado desde el inicio. Buckets `raw`, `features`, `artifacts`.

**No-code.** Salida: dataset elegido y verificado en GCS, tabla de licencia, pregunta del proyecto escrita, proyecto GCP con alerta de presupuesto.

---

## Fase 1 — Exploración y entendimiento del fraude

Fase de código ligero, exploratoria. El objetivo es entendimiento, no producto.

- Cargar los datos desde GCS y caracterizar la prevalencia del fraude (qué tan raro es; típicamente bien por debajo del 1%).
- Caracterizar cómo se distribuye el fraude frente a lo legítimo: montos, horarios, tipos de operación, entidades involucradas.
- Identificar candidatas a señal: qué variables crudas separan parcialmente fraude de no-fraude.
- Verificar la estructura temporal: rango de fechas, densidad de transacciones, viabilidad de un corte temporal con suficiente fraude a ambos lados.

**Código (notebook).** Salida: comprensión documentada del dataset y lista de hipótesis de features para la Fase 2.

---

## Fase 2 — Ingeniería de features de comportamiento

Fase de código, núcleo intelectual del proyecto. Aquí se gana la diferenciación.

- **Modelo de entidades.** Definir las entidades sobre las que se acumula comportamiento (cuenta de origen, tarjeta, comercio destino) y el estado histórico que se mantiene por cada una.
- **Features de desviación.** Por cada transacción, computar cuánto se desvía del patrón histórico de su entidad: razón monto/promedio histórico, tiempo desde la última transacción, velocidad reciente (conteo en ventana), novedad del destino.
- **Ventanas temporales.** Definir las ventanas de agregación (corto, medio, largo plazo) en `FeatureConfig`, versionadas.
- **Regla anti-fuga, no negociable.** Toda agregación usa exclusivamente transacciones anteriores al instante evaluado. Implementar como cómputo causal estricto y cubrir con tests dedicados en `features/tests/`.
- **Paridad por construcción.** `build_features.py` es el único lugar donde una transacción se convierte en vector, importado por igual por entrenamiento y serving.

**Código.** Salida: matriz de features versionada en GCS (`features/feat-v1/`) y suite de tests que verifica ausencia de fuga temporal y determinismo.

---

## Fase 3 — Régimen de evaluación honesto

Fase de código, el diferenciador firmado del proyecto. Se construye la vara de medir **antes** del modelo.

- **Split temporal estricto.** `data/splits.py` define cortes por fecha: entrenamiento sobre el periodo anterior, validación y test sobre periodos posteriores, sin solapamiento. Ninguna entidad ni instante del futuro entra al entrenamiento.
- **Métricas de fraude reales.** `metrics.py` implementa precision-recall (no ROC como cabecera, por el desbalance), curva precision-recall completa, y selección de umbral sobre validación temporal.
- **Curva de costo de negocio.** Traducir el punto de operación a lenguaje de negocio: a tal umbral, se atrapa X% del fraude a costa de revisar Y% de operaciones legítimas. Este es el número que entiende un comprador fintech.
- **Auditoría de fuga.** `leakage_audit.py` corre comprobaciones automáticas que detectan si alguna feature filtra futuro (p. ej. rendimiento sospechosamente perfecto en una feature aislada).
- **Diagnóstico de optimismo.** Reportar también las métricas bajo split aleatorio, etiquetadas explícitamente como optimistas y no desplegables, para cuantificar el gap.

**Código.** Salida: protocolo de evaluación ejecutable que produce la tabla temporal-vs-aleatorio y la curva de costo.

---

## Fase 4 — Modelado

Fase de código, deliberadamente la más corta. El modelo no es el protagonista.

- **Familia del modelo.** Gradient boosting tabular (XGBoost / LightGBM / CatBoost). No red neuronal: peor rendimiento, más lento e innecesariamente complejo para tabular desbalanceado, y con explicabilidad más costosa.
- **Manejo del desbalance.** Ponderación de clase / `scale_pos_weight`, no sobre-muestreo ingenuo que puede introducir fuga si se hace antes del split.
- **Entrenamiento.** Fit sobre el split temporal de entrenamiento. CPU de Colab es suficiente; entrenamiento del orden de minutos.
- **El hallazgo.** Pasar el modelo por el régimen de la Fase 3 y registrar el contraste: número optimista (aleatorio) vs número desplegable (temporal). Ese gap es el resultado central del proyecto.
- **Publicación del bundle.** `export.py` serializa modelo + esquema de features + umbral + metadata a `artifacts/m-v1/`.

**Código.** Salida: bundle versionado en GCS y métricas registradas.

---

## Fase 5 — Servicio end-to-end

Fase de código. Convierte el análisis en producto. Reutiliza el patrón de serving de un sistema GCP ya probado.

- **Endpoint de scoring.** `POST /v1/score` recibe una transacción cruda, la pasa por el mismo `features/` del entrenamiento, aplica el modelo y el umbral, y devuelve score continuo, decisión (`approve` / `review` / `block`) y razón del flag.
- **Explicabilidad por predicción.** `explain.py` devuelve las features que más empujaron la decisión (atribución por feature, nativa del boosting). La razón es parte de la respuesta, no un extra.
- **Runtime singleton.** `model_runtime.py` carga el bundle al arrancar; rechaza arrancar si falta un componente. `/health` y `/version` exponen estado y versiones.
- **Latencia medida.** La respuesta incluye `latency_ms`. El "tiempo real" es un número en pantalla, no un claim.
- **Contrato de respuesta.** Si la transacción no se puede puntuar (campos faltantes, entidad desconocida sin historial), se devuelve un estado explícito, no un error genérico.

**Código.** Salida: servicio desplegado en Cloud Run con endpoint de scoring funcional y latencia instrumentada.

---

## Fase 6 — Dashboard de monitoreo en vivo

Fase de código frontend. Es el componente visual central del proyecto: la cara que se ve en demo y en entrevista. Debe presentar los datos de forma intuitiva e innovadora, no como una tabla plana.

Concepto de diseño (la estructura debe permitir, aunque el detalle se itere después):

- **Stream de transacciones en vivo.** Las transacciones aparecen en tiempo real a medida que se puntúan, codificadas por color según decisión (`approve` verde, `review` ámbar, `block` rojo). El ojo capta el ritmo del fraude entrando al sistema.
- **Score como objeto visual, no número.** Cada transacción muestra su score de riesgo de forma gráfica (medidor, barra, intensidad), no solo un decimal. La decisión se lee de un vistazo.
- **Panel de razones.** Al seleccionar una transacción marcada, el dashboard muestra las features que dispararon el flag, traduciendo la atribución del modelo a lenguaje legible ("monto 12× el promedio histórico de la cuenta", "tercera transacción en 60 segundos").
- **Indicadores agregados.** Tasa de fraude detectado, distribución de decisiones, latencia p50/p99 en vivo, volumen por ventana. KPIs que un risk lead miraría.
- **Control de umbral interactivo.** Un control que mueve el umbral de operación y muestra en vivo el trade-off: más bloqueo atrapa más fraude pero molesta más clientes. Hace tangible la curva de costo de la Fase 3.

Tecnología sugerida: frontend ligero servido same-origin desde Cloud Run (como el demo de un sistema previo), consumiendo el endpoint de scoring y un stream de eventos. Visualización con una librería de charting capaz de actualización en tiempo real. El énfasis es claridad y densidad de información útil, no decoración.

**Código.** Salida: dashboard desplegado, demostrable en vivo, legible en una captura de pantalla.

---

## Fase 7 — Empaquetado y narrativa

Fase mayormente sin código. Captura el retorno; sin ella, el sistema existe pero nadie lo entiende.

- **README final** con resultados reales, la tabla temporal-vs-aleatorio y el gap como protagonista.
- **Model card** (`docs/model_card_m-v1.md`) con métricas, punto de operación y limitaciones explícitas del dato sintético.
- **Declaración de alcance y límites.** Qué se evaluó (scoring por transacción), sobre qué dato (sintético, con sus sesgos), y qué no cubre (no es modelo de producción, no incluye grafos de entidades en v1).
- **Material de difusión.** Guion del post/carrusel con la imagen ancla: el contraste visual entre el número optimista y el desplegable, y una captura del dashboard en vivo.

**No-code.** Salida: repositorio presentable y narrativa lista.

---

## Invariantes de implementación

Romper cualquiera introduce un bug silencioso que solo aparece en producción.

1. **Paridad de features.** Toda transformación transacción→vector vive en `features/`. En cualquier otro módulo se importa; nunca se reimplementa.
2. **Sin fuga temporal.** Ninguna feature usa información posterior al instante de la transacción evaluada. Verificado por tests y por `leakage_audit.py`.
3. **Split temporal obligatorio.** El resultado de cabecera se obtiene siempre con corte temporal. El split aleatorio es solo diagnóstico.
4. **Métricas de desbalance.** La cabecera es precision-recall y costo de negocio, nunca accuracy ni ROC-AUC a secas.
5. **El artefacto es un bundle.** Modelo sin esquema de features ni umbral no es un modelo válido. Los componentes se publican y cargan juntos.
6. **El umbral es un artefacto versionado.** Se selecciona sobre validación temporal con criterio reproducible, no es una constante en el código.
7. **Versiones pineadas y semillas fijas.** `pyproject.toml` pinea las dependencias; las semillas se fijan para reproducibilidad.
8. **Sin datos ni secretos en git.** Los datos viven en GCS; los secretos en variables de entorno o Secret Manager.
9. **Reconstruir features si cambia la config.** La matriz de features está acoplada a `feature_version`. Cambiar cualquier parámetro de `FeatureConfig` requiere incrementar versión y regenerar.

---

## Limitaciones conocidas

- **Dato sintético.** El sistema se entrena y evalúa sobre transacciones sintéticas. Los patrones de fraude sintéticos son más simples y separables que el fraude real; las métricas absolutas no se transfieren a producción. El valor del proyecto es la *metodología* —features causales, evaluación temporal, costo de negocio, serving con explicación— no el número absoluto.
- **Sin grafos de entidades en v1.** El fraude organizado se detecta mejor con relaciones entre entidades (grafos). v1 es tabular por entidad individual; los grafos son una extensión declarada, no parte del alcance inicial.
- **Scoring por transacción, no por sesión.** v1 puntúa transacciones independientes. El fraude de sesión/secuencia queda fuera de alcance.
- **Umbral único.** v1 usa un punto de operación global. La segmentación por tipo de cliente o canal es trabajo futuro.

---

## Setup

```
# Infraestructura GCP (proyecto nuevo, una vez)
export PROJECT_ID=<proyecto-nuevo>
bash infra/00_enable_apis.sh
bash infra/01_buckets.sh
bash infra/02_artifact_registry.sh
bash infra/03_service_accounts.sh

# Ingesta de datos (en Colab: fuente → GCS, sin pasar por local)
# Abrir data/download/colab_ingest.ipynb

# Features + entrenamiento + evaluación (en Colab CPU)
# build_features.py → train.py → evaluate.py

# Deploy del servicio + dashboard
bash infra/04_deploy.sh
```
