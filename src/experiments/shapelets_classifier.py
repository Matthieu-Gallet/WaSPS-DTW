"""
Learning Shapelets wrappers for one-shot classification comparisons.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from optimizer.learning_shapelets import LearningShapelets


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_shapelets_input(samples: List[np.ndarray]) -> np.ndarray:
    """
    Convert list[(T, F)] to LearningShapelets input shape (N, F, T).
    """
    if len(samples) == 0:
        raise ValueError("samples must not be empty")
    arr = np.stack([np.asarray(s, dtype=np.float32) for s in samples], axis=0)  # (N, T, F)
    return np.transpose(arr, (0, 2, 1)).astype(np.float32, copy=False)          # (N, F, T)


def default_shapelets_size_and_len(ts_length: int, num_per_scale: int = 4) -> Dict[int, int]:
    """
    Build a compact default shapelet-length dictionary from time length.
    """
    s1 = max(3, ts_length // 4)
    s2 = max(4, ts_length // 2)
    if s1 == s2:
        return {s1: num_per_scale}
    return {s1: num_per_scale, s2: num_per_scale}


def train_shapelets_classifier(
    train_samples: List[np.ndarray],
    y_train: np.ndarray,
    dist_measure: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    shapelets_size_and_len: Optional[Dict[int, int]] = None,
    shapelets_gamma: float = 1.0,
    shapelets_num_per_scale: int = 4,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[LearningShapelets, Dict]:
    """
    Train a LearningShapelets classifier for a specific distance measure.
    """
    _set_seed(seed)
    x_train = prepare_shapelets_input(train_samples)

    classes = np.sort(np.unique(y_train))
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[int(y)] for y in y_train], dtype=np.int64)

    n_classes = len(classes)
    if shapelets_size_and_len is None:
        shapelets_size_and_len = default_shapelets_size_and_len(
            ts_length=x_train.shape[2],
            num_per_scale=shapelets_num_per_scale,
        )

    clf = LearningShapelets(
        shapelets_size_and_len=shapelets_size_and_len,
        loss_func=torch.nn.CrossEntropyLoss(),
        in_channels=x_train.shape[1],
        num_classes=n_classes,
        dist_measure=dist_measure,
        verbose=verbose,
        to_cuda=False,
        gamma=shapelets_gamma,
    )
    clf.set_optimizer(torch.optim.Adam(clf.model.parameters(), lr=learning_rate))

    start_time = time.time()
    clf.fit(
        x_train,
        y_encoded,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_time = time.time() - start_time

    state = {
        "classes": classes,
        "train_time": train_time,
    }
    return clf, state


def predict_shapelets_classifier(
    clf: LearningShapelets,
    state: Dict,
    test_samples: List[np.ndarray],
    batch_size: int = 256,
) -> np.ndarray:
    """
    Predict class labels with a trained LearningShapelets classifier.
    """
    x_test = prepare_shapelets_input(test_samples)
    logits = clf.predict(x_test, batch_size=batch_size)
    pred_idx = np.argmax(logits, axis=1)
    return state["classes"][pred_idx]
