# Guion de difusión — Dupin

Material para post / carrusel (LinkedIn). La imagen ancla es el **contraste entre
el número fantasioso y el desplegable**; el cierre, una captura del dashboard en
vivo.

## El gancho (slide 1)

> El 90% de los notebooks de detección de fraude reportan AUC de 0.99.
> Casi todos están haciendo trampa. Construí Dupin para mostrar por qué — y cuál
> es el número honesto.

## La historia (carrusel)

**Slide 2 — La trampa.** PaySim anula las transacciones fraudulentas, así que las
columnas de balance codifican la etiqueta. Un modelo ingenuo sobre ellas da
ROC-AUC 0.99. No detecta fraude: lee la respuesta.

**Slide 3 — Dos fugas, no una.** El número inflado mezcla *fuga de etiqueta*
(columnas de balance) y *fuga temporal* (split aleatorio que filtra el futuro al
pasado). Dupin las separa y las mide.

**Slide 4 — La vara honesta.** Split temporal estricto: entrena en el pasado,
evalúa en el futuro, como en producción. Features de comportamiento causales
(sobre la cuenta **receptora**, porque los originadores son de un solo uso),
calculadas solo con información anterior a cada transacción.

**Slide 5 — El número real.**
> 99.8% (fantasía) → **35.3%** de fraude atrapado revisando solo el 1% de
> operaciones, con 75% de precisión. ~78% al 5%. Un tier de auto-block atrapa
> fraude con **cero falsos positivos**.
Modesto comparado con el 99.8%, pero **real y desplegable** — y esa es justo la
diferencia que importa.

**Slide 6 — De extremo a extremo.** Ingesta → features (paridad entrenamiento/
serving por construcción) → evaluación honesta → modelo → serving en Cloud Run →
dashboard en vivo. 43 tests. Todo reproducible.

**Slide 7 — En vivo.** Captura del dashboard: stream de transacciones coloreadas
por decisión, score como medidor, razón legible del flag, y un control de umbral
que hace tangible el trade-off recall vs revisión.

## Cierre

> El clasificador es casi un commodity. El valor está en cómo lo evalúas: si no
> mides con split temporal, tu 0.99 es ficción. Dupin es la metodología, no el
> número.
>
> Código y demo en vivo 👇

## Activos visuales

- `docs/images/curva-pr-temporal-vs-aleatorio.png` — el optimismo del split aleatorio.
- `docs/images/envolvente-recall-presupuesto.png` — recall vs presupuesto (el techo honesto).
- Captura del dashboard en vivo (review/block sobre fraude real) → desde
  `https://dupin-705834513207.us-central1.run.app`.

## Hashtags sugeridos

`#MachineLearning #FraudDetection #MLOps #DataScience #GCP #HonestEvaluation`
