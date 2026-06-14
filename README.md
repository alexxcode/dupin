# Dupin — Real-Time Transaction Fraud Scoring

Sistema de scoring de riesgo de fraude transaccional en tiempo real. Dada una
transacción dentro de un flujo de operaciones, emite un score continuo, una
decisión (`approve` / `review` / `block`) y la razón explícita del flag, con
latencia medida bajo carga.

**Estado:** prototipo v0.1 · modelo `m-v1` · features `feat-v1`
**Stack:** GCP (GCS + Colab + Cloud Run) · gradient boosting tabular · FastAPI · dashboard en vivo

> El eje del proyecto **no es el clasificador**, sino el **régimen de evaluación
> honesto**. El resultado de cabecera es el *gap* entre el número optimista y el
> desplegable, descompuesto en sus dos causas: **fuga de etiqueta** y **fuga
> temporal**.

---

## Pregunta del proyecto

> Dada una transacción de dinero móvil dentro de un flujo temporal, ¿se puede
> predecir si es fraudulenta usando solo información de comportamiento disponible
> *antes* del instante de la transacción, y cuánto se degrada ese rendimiento al
> evaluar con split temporal honesto en lugar del split aleatorio optimista —y
> cuánto del rendimiento aparente de los enfoques ingenuos es fuga de etiqueta de
> las columnas de balance?

**Resultado que se cuenta:** el gap entre el número ingenuo/optimista y el
honesto/desplegable, descompuesto en (1) fuga de etiqueta y (2) fuga temporal.

---

## Dataset, licencia y atribución

Construido sobre **PaySim**, un simulador público de transacciones de dinero
móvil. Datos sintéticos: la metodología es el valor, no el número absoluto.

- **Atribución (CC BY):** Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016).
  *PaySim: A financial mobile money simulator for fraud detection.* En *The 28th
  European Modeling and Simulation Symposium (EMSS)*, Larnaca, Cyprus.
- **Fuente:** Kaggle — [`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Licencia del dataset:** CC BY-SA 4.0.

**ShareAlike, en claro:** el SA aplica a redistribuciones del dataset o derivados
del dato. **No** aplica a este código, al modelo entrenado ni al dashboard, que
lo consumen pero no son el dato. Por eso: el CSV de PaySim **nunca** entra al
repo (vive en GCS + Kaggle), se atribuye la fuente, y el código va bajo licencia
MIT. Ver [`.gitignore`](.gitignore) (Invariante 8: sin datos ni secretos en git).

### La trampa de las columnas de balance

La documentación de PaySim advierte que las transacciones fraudulentas son
*anuladas*, por lo que `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest` y
`newbalanceDest` **codifican el etiquetado**. Usarlas crudas da AUC ~0.99 que es
**fuga de etiqueta**, no detección. Dupin las prohíbe como features crudas, lo
demuestra en la Fase 1 y lo audita en la Fase 3. Ese error —común en notebooks de
Kaggle— es uno de los dos ejes del hallazgo central.

---

## Estructura del repositorio

```
dupin/
├── data/
│   ├── download/colab_ingest.ipynb   # ingesta Kaggle→GCS (sin pasar por local)
│   ├── schema.py                     # esquema tipado de la transacción cruda
│   └── splits.py                     # cortes temporales (Fase 3)
├── notebooks/                        # 01 exploración · 02 features · 03 eval · 04 train
├── features/                         # Fase 2 — transacción→vector, sin fuga (+ tests)
├── training/                         # Fase 4 — train + model + bundle (+ tests)
├── evaluation/                       # Fase 3 — métricas, splits, leakage_audit (+ tests)
├── serving/                          # Fase 5 — FastAPI, runtime, explain, Dockerfile (+ tests)
├── infra/                            # 00-04 scripts GCP (apis, buckets, AR, SA, deploy)
├── docs/                             # findings por fase + model card
├── cloudbuild.yaml · config.example.yaml · pyproject.toml
```

---

## Estado por fase

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0 | Dataset, licencia, pregunta, GCP | ✅ Completa |
| 1 | Exploración y entendimiento del fraude | ✅ Completa — ver [docs/phase1_findings.md](docs/phase1_findings.md) |
| 2 | Features de comportamiento | ✅ Completa — matriz en `gs://dupin-dupin-features/feat-v1/` (2.77M filas) |
| 3 | Régimen de evaluación honesto | ✅ Completa — resultado en [docs/phase3_findings.md](docs/phase3_findings.md) (recall 99.8% → 20.7%) |
| 4 | Modelado | ✅ Completa — LightGBM, bundle `m-v1` · ver [docs/phase4_findings.md](docs/phase4_findings.md) (recall 35.3% @1%, 78% @5%) |
| 5 | Serving end-to-end | ✅ **Desplegado en Cloud Run** · FastAPI `/v1/score` · 41/41 tests |
| 6 | Dashboard en vivo | ⏳ |
| 7 | Empaquetado y narrativa | ⏳ |

---

## Resultados (m-v1) — el gap honesto

El protagonista no es el AUC, sino el **gap descompuesto**: cuánto del rendimiento
aparente era fuga. Punto de operación: **revisar ≤1% de operaciones**.

| Configuración | Recall | Precision | Nota |
|---|---|---|---|
| Balance crudo + split aleatorio | **99.8%** | — | fantasía: fuga de etiqueta + temporal |
| Honesto + split temporal (desplegable) | **35.3%** | 75.4% | el número real, a presupuesto completo |
| — tier auto-block | 1.74% | **100%** | bloqueo sin falsos positivos |

Envolvente honesta: ~54% recall al 2% de revisión, ~78% al 5%. PR-AUC temporal
0.594 (≈28× sobre el azar). **De 99.8% fantasioso a 35.3% honesto** — ese contraste,
descompuesto en fuga de etiqueta (columnas de balance) y fuga temporal (split), es
el resultado. Detalle: [docs/phase3_findings.md](docs/phase3_findings.md) ·
[docs/phase4_findings.md](docs/phase4_findings.md).

---

## API de scoring

**En vivo:** `https://dupin-705834513207.us-central1.run.app` (Cloud Run, escala a cero).

`POST /v1/score` — transacción cruda → score + decisión + razón legible + latencia:

```bash
curl -X POST localhost:8080/v1/score -H 'Content-Type: application/json' -d '{
  "step": 500, "type": "TRANSFER", "amount": 181000.0,
  "nameOrig": "C840083671", "nameDest": "C38997010"
}'
# → {"score":0.97,"decision":"block","scorable":true,
#    "reasons":[{"feature":"dest_amount_ratio","message":"Monto 12.4× el promedio histórico del receptor",...}],
#    "latency_ms":3.1,"model_version":"m-v1","feature_version":"feat-v1"}
```

Otros: `GET /health` (estado + modelo cargado), `GET /version`, `GET /docs` (OpenAPI).
El serving usa el **mismo `features/`** del entrenamiento (paridad) y rechaza
arrancar si falta un componente del bundle.

```bash
# Local
export DUPIN_BUNDLE_URI=gs://dupin-dupin-artifacts/m-v1   # o ruta local
uvicorn serving.app:app --reload
```

---

## Setup

```bash
cp config.example.yaml config.yaml          # configuración

# Plano offline (Colab): ingesta → features → evaluación → modelo
#   data/download/colab_ingest.ipynb     → PaySim a gs://dupin-dupin-raw/
#   notebooks/01_fraud_exploration.ipynb → caracterización (Fase 1)
#   notebooks/02_build_features.ipynb    → matriz feat-v1 (Fase 2)
#   notebooks/03_evaluate.ipynb          → gap honesto (Fase 3)
#   notebooks/04_train.ipynb             → bundle m-v1 (Fase 4)

# Plano online (Cloud Run): build + deploy del serving
export PROJECT_ID=dupin-dupin
bash infra/00_enable_apis.sh
bash infra/01_buckets.sh
bash infra/02_artifact_registry.sh
bash infra/03_service_accounts.sh
bash infra/04_deploy.sh                       # verifica el bundle y despliega

# Tests
python -m pytest -q
```

GCP: proyecto `dupin-dupin`, alerta de presupuesto activa.
