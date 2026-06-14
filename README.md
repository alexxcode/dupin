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
├── notebooks/
│   └── 01_fraud_exploration.ipynb    # Fase 1 — exploración y entendimiento
├── features/                         # Fase 2 — transacción→vector, sin fuga
├── training/                         # Fase 4 — split temporal → fit → bundle
├── evaluation/                       # Fase 3 — PR, costo, auditoría de fuga
├── serving/                          # Fase 5/6 — FastAPI + dashboard
├── infra/                            # scripts GCP
├── docs/                             # model card
├── config.example.yaml
└── pyproject.toml
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
| 5 | Serving end-to-end | ⏳ |
| 6 | Dashboard en vivo | ⏳ |
| 7 | Empaquetado y narrativa | ⏳ |

---

## Setup

```bash
# Configuración (copiar plantilla y ajustar)
cp config.example.yaml config.yaml

# Fase 1 (en Colab): ingesta + exploración
#   1. data/download/colab_ingest.ipynb   → PaySim a gs://dupin-dupin-raw/
#   2. notebooks/01_fraud_exploration.ipynb → caracterización del fraude
```

GCP: proyecto `dupin-dupin`, alerta de presupuesto activa.
