# Fase 4 — Modelo final (LightGBM) y bundle `m-v1`

Producido por [`notebooks/04_train.ipynb`](../notebooks/04_train.ipynb) sobre la
matriz `feat-v1`. Bundle publicado en `gs://dupin-dupin-artifacts/m-v1/`.

## El número honesto, bien medido

PR-AUC test = **0.594** (prevalencia 2.14% → **~28× sobre el azar**). El modelo
aprendió señal real.

| Punto de operación | Recall | Precision | Review rate |
|---|---|---|---|
| Desplegado (umbral val→test, conservador) | 21.0% | 95.4% | 0.47% |
| **Envolvente al 1% (capacidad real)** | **35.3%** | 75.4% | 1.00% |
| Auto-block (tier alta precisión) | 1.74% | **100%** | 0.04% |

Curva envolvente sobre el test temporal: ~54% recall al 2% de revisión, ~78% al 5%.

**De la fantasía a la realidad (recall):** 99.8% (balance crudo + aleatorio) →
**35.3%** (honesto + temporal, al presupuesto completo). El número desplegable es
modesto pero **real y útil**: a 1% de revisión atrapa 1 de cada 3 fraudes con 75%
de precisión, y el tier de auto-block elimina fraude con cero falsos positivos.

## Desplegado vs envolvente — la transferencia de umbral

El punto desplegado (21%) gasta solo 0.47% del presupuesto de 1%: el umbral
fijado sobre validación (prevalencia 3.96%) queda demasiado estricto para el test
(2.14%). La **capacidad real** del modelo es la envolvente (35.3% al 1%). En
producción esto se recupera **recalibrando el umbral** sobre datos recientes para
mantener el review rate objetivo — práctica estándar de monitoreo de drift, que el
dashboard (Fase 6) puede exponer. El bundle guarda el umbral conservador como
default seguro; la envolvente documenta el techo.

## Importancia de features — valida la tesis del proyecto

| # | Feature | Importancia |
|---|---|---|
| 1 | log_amount | 4738 |
| 2 | dest_amt_sum_168h | 3204 |
| 3 | dest_amount_ratio | 3074 |
| 4 | dest_recency | 2824 |
| 5 | dest_amount_z | 2731 |
| 6 | hour | 2649 |
| 7 | dest_amt_sum_24h | 2212 |
| 8 | dest_prior_count | 1513 |
| 9 | dest_cnt_168h | 1014 |
| 10 | dest_cnt_24h | 647 |
| 11 | type_transfer | 525 |
| 12 | dest_is_new | 51 |
| 13 | orig_prior_count | 18 |

Las features de comportamiento del **receptor** dominan (velocidad 168h, razón de
monto, recencia, z-score) — exactamente el pivote a `nameDest` decidido en la
Fase 1. `orig_prior_count` es el último (18), confirmando que los originadores de
un solo uso son peso muerto (candidato a quitar en feat-v2).

## Bundle `m-v1`

`gs://dupin-dupin-artifacts/m-v1/`: `model.joblib` + `manifest.json` (esquema de
features, umbrales review=0.9984 / block=0.9999, métricas, metadata). `load`
rechaza arrancar si falta cualquier componente. El serving (Fase 5) lo descarga,
usa el mismo `features/`, y aplica `decide()` compartida.

## Estado

Fase 4 **completa**. Modelo final entrenado, evaluado bajo el régimen honesto, y
bundle publicado. Pendiente para feat-v2/m-v2: quitar `orig_prior_count`,
recalibración de umbral automática.
