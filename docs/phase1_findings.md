# Fase 1 — Hallazgos de la exploración de PaySim

Entregable de la Fase 1: comprensión documentada del dataset + hipótesis de
features para la Fase 2. Números obtenidos ejecutando
[`notebooks/01_fraud_exploration.ipynb`](../notebooks/01_fraud_exploration.ipynb)
sobre `gs://dupin-dupin-raw/raw/paysim/` (6.362.620 filas).

## Resumen numérico

| Métrica | Valor |
|---|---|
| Transacciones totales | 6.362.620 |
| Fraudes | 8.213 (**0.1291 %** · 1 por cada 774) |
| Flag nativo `isFlaggedFraud` | atrapa 16/8.213 = **0.19 %** (baseline inútil) |
| Tipos con fraude | **solo TRANSFER y CASH_OUT** (0 en el resto) |
| Demo de fuga (balance crudo + split aleatorio) | ROC-AUC **0.9879** · PR-AUC **0.6372** |

## Fraude por tipo de operación

| tipo | legit | fraude | fraud_rate |
|---|---|---|---|
| TRANSFER | 528.812 | 4.097 | 0.7688 % |
| CASH_OUT | 2.233.384 | 4.116 | 0.1840 % |
| CASH_IN | 1.399.284 | 0 | 0 % |
| DEBIT | 41.432 | 0 | 0 % |
| PAYMENT | 2.151.495 | 0 | 0 % |

**Implicación:** restringir el problema a `{TRANSFER, CASH_OUT}` elimina 3.59M
filas irrelevantes y eleva la tasa base de 0.13 % a ~0.30 %. Filtro estructural
para Fase 2/3.

## Fuga de etiqueta — confirmada

PaySim anula las transacciones fraudulentas; las columnas de balance codifican el
etiquetado. Medias por clase (Sección 6):

| columna | legit | fraude |
|---|---|---|
| errBalanceOrig | 201.339 | 10.692 |
| errBalanceDest | 54.692 | 732.509 |

Un modelo trivial sobre estas columnas + split aleatorio da ROC-AUC 0.99. Es
**trampa, no detección**. → `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`,
`newbalanceDest` quedan **prohibidas como features crudas** (ver
[`data/schema.py`](../data/schema.py) → `LEAKED_BALANCE_COLUMNS`). La Fase 3
descompone el gap optimista en sus dos causas (fuga de etiqueta + fuga temporal).

## ⚠️ Hallazgo 1 — Los originadores son de un solo uso

- `nameOrig` únicos: **6.353.307 / 6.362.620** (99.85 %).
- Reincidencia por originador: media **1.0015**, máximo **3**.
- `nameDest` únicos: 2.722.362 / 6.362.620 → los destinos sí repiten (~2.3 tx c/u).
- Comercios (`nameDest` con prefijo `M`): 33.8 % de los destinos; **0 fraudes** van
  a comercio. El fraude es siempre C→C.

**Consecuencia de diseño (reescribe las hipótesis de features):** no existe
historia acumulable por `nameOrig` — casi ningún originador aparece dos veces. Las
features de comportamiento (velocidad, recencia, desviación de monto, novedad)
**deben acumularse sobre `nameDest`**, la cuenta receptora, no sobre el
originador.

## ⚠️ Hallazgo 2 — El volumen está muy desbalanceado en el tiempo

Corte candidato en step 480 (día 20):

| periodo | filas | fraude | tasa |
|---|---|---|---|
| train `[0, 480)` | 6.039.437 (95 %) | 5.343 | 0.0885 % |
| test `[480, 744)` | 323.183 (5 %) | 2.870 | 0.8880 % |

El split temporal **es viable** (fraude de sobra a ambos lados), pero:
- 95 % del volumen cae en train, solo 5 % en test.
- La tasa de fraude del test es **10× la del train** → posible drift del simulador
  o artefacto de la distribución de `step`.

**Consecuencia de diseño:** el punto de corte debe reconsiderarse en la Fase 3.
Opciones a evaluar: mover el corte para equilibrar volumen, o aceptar el régimen
de cola documentando explícitamente su no-representatividad. No bloquea Fase 2.

## Hipótesis de features para la Fase 2 (actualizadas)

Todas causales — solo información **anterior** al instante de la transacción.
**Pivote clave: acumular sobre `nameDest`, no `nameOrig`.**

1. **Razón de monto:** `amount` / promedio histórico de montos recibidos por el `nameDest`.
2. **Recencia:** steps desde la última transacción recibida por el `nameDest`.
3. **Velocidad:** conteo de transacciones hacia el `nameDest` en ventanas (corto/medio plazo).
4. **Novedad del par:** ¿primera vez que este `nameDest` recibe (o de este `nameOrig`)?
5. **Desviación de monto:** z-score del monto vs. historia de montos del `nameDest`.
6. **Señal de tipo:** indicador TRANSFER/CASH_OUT cruzado con la desviación (el fraude solo vive ahí).
7. **(Balance, honesta):** errores de balance solo si pasan la auditoría de fuga; por defecto **excluidas en v1**.

Estas pasan a `features/config.py` (`FeatureConfig`: ventanas, entidades,
defaults) y `features/build_features.py`.

## Estado

Fase 1 **completa**. Dato caracterizado, fuga confirmada, dos restricciones de
diseño identificadas (entidad receptora · corte temporal), hipótesis de features
listas para Fase 2.
