"""Construcción del modelo final — gradient boosting tabular (LightGBM).

No red neuronal: en tabular desbalanceado los árboles con boosting ganan en
rendimiento, velocidad e interpretabilidad. El desbalance se maneja con
`scale_pos_weight`, no con sobre-muestreo (que puede introducir fuga si se hace
antes del split). Semilla fija para reproducibilidad (Invariante 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    model_version: str = "m-v1"
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 64
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    seed: int = 42
    extra: dict = field(default_factory=dict)


DEFAULT_MODEL_CONFIG = ModelConfig()


def make_lightgbm(config: ModelConfig = DEFAULT_MODEL_CONFIG, scale_pos_weight: float = 1.0):
    """Crea un LightGBMClassifier sin entrenar. Import perezoso: lightgbm solo se
    necesita en el plano offline (Colab), no en los tests de bundle/umbral."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=config.seed,
        n_jobs=-1,
        verbose=-1,
        **config.extra,
    )
