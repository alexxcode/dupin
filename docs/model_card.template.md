# Model Card — Dupin `<model_version>`

Plantilla de tarjeta de modelo. Copiar a `model_card_<model_version>.md` y rellenar
con métricas reales tras evaluar bajo el régimen temporal (Fase 3).

| | |
|---|---|
| **Modelo** | `<model_version>` · `<familia>` |
| **Features** | `<feature_version>` · `<n>` features causales |
| **Tarea** | Clasificación binaria por transacción: P(fraude) |
| **Dataset** | `<dataset>` · `<licencia>` |
| **Artefacto** | `gs://<bucket>-artifacts/<model_version>/` |
| **Serving** | `<plataforma>` · `POST /v1/score` |

## Uso previsto
- Para qué sirve / para qué NO sirve / usuarios.

## Datos
- Fuente, licencia, prevalencia, superficie de scoring, fuga evitada.

## Features
- Lista, regla causal, entidad de acumulación, top de importancia.

## Evaluación
- Split temporal (cortes), punto de operación de cabecera.
- Tabla: desplegado vs envolvente (recall / precision / review rate).
- PR-AUC temporal. **El gap honesto** (optimista vs desplegable), descompuesto.

## Umbrales y decisión
- review / block, regla de decisión, transferencia de umbral.

## Explicabilidad
- Método de atribución y formato de la razón.

## Limitaciones
- Dato sintético · sin grafos · por transacción · umbral único · cold-start.

## Ética y riesgos
- PII, consejo financiero, sesgo del dato.

## Reproducibilidad
- Semillas, dependencias pineadas, bundle versionado, tests.
