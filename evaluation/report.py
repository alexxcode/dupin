"""report — formatea el reporte de evaluación a tabla legible y JSON de resumen."""
from __future__ import annotations

import json


def format_table(report: dict) -> str:
    """Tabla temporal-vs-aleatorio en markdown. El gap es el protagonista."""
    t = report["temporal"]
    r = report["random"]
    top, rop = t["operating_point"], r["operating_point"]
    budget = report["budget"]
    lines = [
        f"Punto de operación: revisar ≤ {budget:.1%} de operaciones "
        f"(umbral fijado sobre validación).",
        "",
        "| Régimen | PR-AUC | ROC-AUC | Recall@budget | Precision | Review rate |",
        "|---|---|---|---|---|---|",
        _row("Aleatorio (optimista, NO desplegable)", r, rop),
        _row("**Temporal (honesto, desplegable)**", t, top),
        "",
        f"**Gap por fuga temporal:** PR-AUC −{report['gap']['pr_auc']:.4f} · "
        f"Recall −{report['gap']['recall']:.4f}",
    ]
    return "\n".join(lines)


def _row(name: str, regime: dict, op: dict) -> str:
    return (
        f"| {name} | {regime['pr_auc']:.4f} | {regime['roc_auc']:.4f} | "
        f"{op['recall']:.4f} | {op['precision']:.4f} | {op['review_rate']:.4f} |"
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
