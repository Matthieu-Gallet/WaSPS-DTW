"""
Simple LSTM baseline for barycenter-based classification on raw samples.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stack_samples(samples: List[np.ndarray]) -> np.ndarray:
    if len(samples) == 0:
        raise ValueError("samples must not be empty")
    return np.stack([s.astype(np.float32, copy=False) for s in samples], axis=0)


def _compute_standardization(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def train_lstm_classifier(
    X_train_raw: List[np.ndarray],
    y_train: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[_LSTMClassifier, Dict]:
    """
    Train an LSTM encoder on raw time-series samples.
    """
    _set_seed(seed)

    X_train = _stack_samples(X_train_raw)
    mean, std = _compute_standardization(X_train)
    X_train = (X_train - mean) / std

    classes = np.sort(np.unique(y_train))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_train], dtype=np.int64)

    x_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_encoded)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _LSTMClassifier(
        input_size=X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(classes),
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_obs = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_n = yb.shape[0]
            running_loss += float(loss.item()) * batch_n
            n_obs += batch_n

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  LSTM epoch {epoch + 1:3d}: loss={running_loss / max(n_obs, 1):.6f}")

    train_time = time.time() - start_time

    state = {
        "mean": mean,
        "std": std,
        "classes": classes,
        "device": device,
        "train_time": train_time,
    }
    return model, state


def _encode_lstm_samples(
    model: _LSTMClassifier,
    state: Dict,
    X_raw: List[np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode raw samples into latent vectors using the trained LSTM."""
    X = _stack_samples(X_raw)
    X = (X - state["mean"]) / state["std"]
    x_tensor = torch.from_numpy(X)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(state["device"])
            emb = model.encode(xb)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def compute_lstm_barycenters(
    model: _LSTMClassifier,
    state: Dict,
    X_train_raw: List[np.ndarray],
    y_train: np.ndarray,
    batch_size: int = 64,
) -> Dict[int, np.ndarray]:
    """
    Compute one raw-space prototype per class from latent barycenter (medoid).
    """
    embeddings = _encode_lstm_samples(model, state, X_train_raw, batch_size=batch_size)
    x_train = _stack_samples(X_train_raw)
    barycenters: Dict[int, np.ndarray] = {}
    for cls in np.sort(np.unique(y_train)):
        class_mask = (y_train == cls)
        if not np.any(class_mask):
            continue
        emb_cls = embeddings[class_mask]
        raw_cls = x_train[class_mask]
        center = emb_cls.mean(axis=0, keepdims=True)
        sq_dists = np.sum((emb_cls - center) ** 2, axis=1)
        medoid_idx = int(np.argmin(sq_dists))
        barycenters[int(cls)] = raw_cls[medoid_idx]
    return barycenters


def classify_by_lstm_barycenters(
    model: _LSTMClassifier,
    state: Dict,
    barycenters: Dict[int, np.ndarray],
    X_raw: List[np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Classify samples by nearest encoded barycenter (Euclidean distance).
    """
    embeddings = _encode_lstm_samples(model, state, X_raw, batch_size=batch_size)
    class_labels = np.array(sorted(barycenters.keys()), dtype=np.int64)
    centers_raw = [np.asarray(barycenters[int(c)], dtype=np.float32) for c in class_labels]
    centers_emb = _encode_lstm_samples(model, state, centers_raw, batch_size=batch_size)
    centers = np.asarray(centers_emb, dtype=np.float64)
    sq_dists = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(sq_dists, axis=1)
    return class_labels[nearest]
