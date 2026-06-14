"""report — formatea el reporte de evaluación a tabla legible y JSON de resumen."""
from __future__ import annotations

import json


def format_table(report: dict) -> str:
    """Tabla temporal-vs-aleatorio en markdown. El gap de recall es el protagonista.

    Cabecera = recall@budget (comparable entre regímenes, al mismo punto de
    operación). El PR-AUC se muestra junto a la prevalencia del test porque NO es
    comparable entre regímenes de distinta prevalencia.
    """
    t = report["temporal"]
    r = report["random"]
    top, rop = t["operating_point"], r["operating_point"]
    budget = report["budget"]
    lines = [
        f"Punto de operación: revisar ≤ {budget:.1%} de operaciones "
        f"(umbral fijado sobre validación).",
        "",
        "| Régimen | Recall@budget | Precision | Review rate | PR-AUC | prev. test |",
        "|---|---|---|---|---|---|",
        _row("Aleatorio (optimista, NO desplegable)", r, rop),
        _row("**Temporal (honesto, desplegable)**", t, top),
        "",
        f"**Gap por fuga temporal (recall@budget): −{report['gap']['recall']:.4f}** "
        f"— el split aleatorio sobreestima el fraude atrapado.",
        f"_PR-AUC no comparable entre regímenes (prevalencia test "
        f"{r['test_prevalence']:.4f} vs {t['test_prevalence']:.4f})._",
    ]
    return "\n".join(lines)


def _row(name: str, regime: dict, op: dict) -> str:
    return (
        f"| {name} | {op['recall']:.4f} | {op['precision']:.4f} | "
        f"{op['review_rate']:.4f} | {regime['pr_auc']:.4f} | "
        f"{regime['test_prevalence']:.4f} |"
    )


def format_split(report: dict) -> str:
    """Tabla de conteos por segmento (verificación de viabilidad del split)."""
    sr = report["split_report"]
    lines = [
        "| Segmento | Filas | Fraude | Tasa | Steps |",
        "|---|---|---|---|---|",
    ]
    for seg in ("train", "val", "test"):
        s = sr[seg]
        lines.append(
            f"| {seg} | {s['rows']:,} | {s['fraud']:,} | {s['fraud_rate']:.4%} | "
            f"[{s['step_min']}, {s['step_max']}] |"
        )
    return "\n".join(lines)


def to_json(report: dict, indent: int = 2) -> str:
    # Ignora claves privadas (p. ej. `_scores`, no serializables).
    serializable = {k: v for k, v in report.items() if not k.startswith("_")}
    return json.dumps(serializable, indent=indent)


def save_json(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(report))
