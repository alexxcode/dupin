"""Esquema tipado de la transacción cruda de PaySim.

Fuente: PaySim — Lopez-Rojas, Elmir & Axelsson (2016), "PaySim: A financial
mobile money simulator for fraud detection", EMSS. CC BY-SA 4.0.
Vía Kaggle: ealaxi/paysim1.

Esta es la ÚNICA definición canónica de las columnas crudas. Importar desde
aquí; nunca redefinir en otro módulo (paridad por construcción).
"""
from __future__ import annotations

from enum import Enum


class TxType(str, Enum):
    """Tipos de operación en PaySim."""

    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


# Columnas crudas, en el orden del CSV de PaySim.
RAW_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",   # nota: PaySim usa "Org" (origen) aquí...
    "newbalanceOrig",  # ...y "Orig" aquí. No es typo: respetar tal cual el CSV.
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# dtypes para carga eficiente y determinista (Invariante: dtype/shape estables).
RAW_DTYPES: dict[str, str] = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# Etiqueta objetivo.
LABEL_COLUMN: str = "isFraud"

# Eje temporal. 1 step = 1 hora simulada; 744 steps = 30 días.
# Indispensable para el split temporal honesto (Fase 3).
TIME_COLUMN: str = "step"

# Identificadores de entidad. Base de las features de comportamiento (Fase 2).
ENTITY_ORIG: str = "nameOrig"
ENTITY_DEST: str = "nameDest"
ENTITY_COLUMNS: list[str] = [ENTITY_ORIG, ENTITY_DEST]

# ─────────────────────────────────────────────────────────────────────────────
# CRÍTICO — FUGA DE ETIQUETA.
# PaySim ANULA las transacciones marcadas como fraude, por lo que estas cuatro
# columnas de balance codifican el propio etiquetado. Usarlas crudas como
# features produce un rendimiento casi perfecto (~0.99) que es trampa, no
# detección. Prohibidas como features crudas. La Fase 1 lo demuestra
# empíricamente y la Fase 3 (leakage_audit.py) lo audita de forma automática.
# ─────────────────────────────────────────────────────────────────────────────
LEAKED_BALANCE_COLUMNS: list[str] = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Columna de flag heurístico nativo de PaySim (regla simple del simulador, no
# nuestra predicción). Se reporta como baseline, no se usa como feature.
FLAG_COLUMN: str = "isFlaggedFraud"
