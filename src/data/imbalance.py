from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight


def smote_sampler(sampling_strategy: float = 0.1, seed: int = 42) -> SMOTE:
    return SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=seed,
        k_neighbors=5,
    )


def undersampler(sampling_strategy: float = 0.1, seed: int = 42) -> RandomUnderSampler:
    return RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=seed,
    )


def class_weight_dict(y: Any) -> dict[int, float]:
    import numpy as np

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights, strict=True))


def get_strategy(name: str, config: dict[str, Any], seed: int = 42):
    if name == "smote":
        return smote_sampler(config.get("sampling_strategy", 0.1), seed)
    elif name == "undersample":
        return undersampler(config.get("sampling_strategy", 0.1), seed)
    elif name in ("class_weight", "none"):
        return None
    else:
        raise ValueError(f"Unknown imbalance strategy: {name}")
