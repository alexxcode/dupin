# Fase 3 — Resultado central: el gap honesto, descompuesto

Producido por [`notebooks/03_evaluate.ipynb`](../notebooks/03_evaluate.ipynb)
sobre la matriz `feat-v1` (2.77M filas). Baseline: `HistGradientBoostingClassifier`
con `scale_pos_weight`. Punto de operación: **recall @ revisar ≤1% de operaciones**.

## La tabla 2×2

| Configuración | Recall@1% | Review rate | PR-AUC | prev. test |
|---|---|---|---|---|
| Balance crudo + ALEATORIO (ingenuo, fuga×2) | **99.8%** | 0.90% | 0.998 | 0.30% |
| Balance crudo + temporal | 21.8% | 0.47% | 0.9997 | 2.14% |
| Honesto + ALEATORIO (optimista) | **61.6%** | 0.90% | 0.485 | 0.30% |
| **Honesto + temporal (DESPLEGABLE)** | **20.7%** | 0.46% | 0.591 | 2.14% |

Viabilidad del split (scope TRANSFER+CASH_OUT):

| Segmento | Filas | Fraude | Tasa | Steps |
|---|---|---|---|---|
| train | 2.623.886 | 4.853 | 0.185% | [1, 431] |
| val | 12.363 | 490 | 3.963% | [432, 479] |
| test | 134.160 | 2.870 | 2.139% | [480, 743] |

## El titular

**De la fantasía a la realidad: recall 99.8% → 20.7%.** El modelo ingenuo (columnas
de balance + split aleatorio) parece atrapar el 99.8% del fraude revisando <1% de
operaciones. El modelo honesto y desplegable (features causales + split temporal)
atrapa el **20.7%**. Todo lo demás era fuga.

## La sutileza metodológica (esto ES el proyecto)

**El PR-AUC NO es comparable entre el régimen temporal y el aleatorio**, porque sus
test tienen prevalencia distinta:
- test aleatorio: prevalencia global **0.30%**.
- test temporal: la cola densa en fraude, **2.14%** (7× mayor).

El PR-AUC de un clasificador aleatorio es igual a la prevalencia, así que el test
temporal arranca con un piso 7× más alto. Por eso el PR-AUC honesto-temporal
(0.591) sale *mayor* que el aleatorio (0.485) — **no porque temporal sea mejor, sino
por el desbalance de prevalencia**. Tomar ese PR-AUC como "el temporal es mejor"
sería justo el tipo de error que este proyecto existe para cazar.

**Conclusión: cada eje se mide con la métrica que sí es válida ahí.**

- **Eje A — fuga de ETIQUETA → PR-AUC (ranking).** A split fijo (misma prevalencia),
  el balance rankea casi perfecto: **0.998 (aleatorio), 0.9997 (temporal)** vs honesto
  **0.485 / 0.591**. La fuga es real y enorme — el AUC fantasioso de los notebooks de
  Kaggle, reproducido y aislado.
- **Eje B — fuga TEMPORAL → recall@budget (punto de operación).** Al mismo
  presupuesto de revisión, el split aleatorio promete **61.6%** de recall; en
  producción (temporal) se obtiene **20.7%**. El split aleatorio **sobreestima el
  recall ~3×** (gap −0.409).

## Hallazgo secundario — el ranking perfecto no se despliega solo

El modelo con fuga de balance tiene PR-AUC temporal 0.9997 (ranking casi perfecto),
pero su recall@1% temporal cae a **21.8%** — casi igual al honesto (20.7%). Su
umbral elegido en validación no transfiere al test por el shift de prevalencia
(val 3.96% vs test 2.14%). Lección: un solo número (PR-AUC alto) no garantiza un
sistema desplegable; el punto de operación importa tanto como el ranking.

## Auditoría de fuga

- **Features honestas: PASA.** Ninguna feature aislada alcanza AUC sospechoso
  (≥0.98). La más fuerte: `dest_recency` 0.817, `dest_cnt_168h` 0.805,
  `dest_amt_sum_168h` 0.805 → **la recencia y la velocidad del receptor son las
  señales individuales más fuertes**.
- **Control positivo (balance):** `errBalanceOrig` 0.947, `oldbalanceOrg` 0.928 —
  elevadas pero <0.98 individualmente. Confirma que **la fuga de etiqueta de PaySim
  es multi-columna**: no la caza una auditoría de feature aislada, sino el modelo
  combinado (eje A arriba). La auditoría sirve para fugas de una sola feature; la
  multi-columna se ve en el contraste de PR-AUC.

## Estado

Fase 3 **completa**. El régimen produce el gap descompuesto y deja claro qué
métrica vale para cada eje. El número honesto+temporal (recall 20.7% @1%, PR-AUC
0.591) es el único desplegable y la base de la model card (Fase 7). La Fase 4
cambia el baseline por XGBoost/LightGBM y lo pasa por este mismo régimen.
