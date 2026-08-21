"""Small feed-forward neural network implemented entirely with NumPy."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> 'StandardScaler':
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('Scaler has not been fitted.')
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('Scaler has not been fitted.')
        return np.asarray(x, dtype=np.float64) * self.std_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('Scaler has not been fitted.')
        return {'mean': self.mean_.tolist(), 'std': self.std_.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'StandardScaler':
        obj = cls()
        obj.mean_ = np.array(payload['mean'], dtype=np.float64)
        obj.std_ = np.array(payload['std'], dtype=np.float64)
        return obj


class DenseNetwork:
    """One hidden-layer MLP trained with mini-batch gradient descent."""

    def __init__(self, input_size: int, hidden_size: int = 48, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_size), size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_size), size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        a1 = self.relu(z1)
        y = a1 @ self.w2 + self.b2
        return y, (z1, a1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].ravel()

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 120,
        learning_rate: float = 0.0015,
        batch_size: int = 32,
        patience: int = 18,
        seed: int = 42,
    ) -> dict[str, list[float]]:
        rng = np.random.default_rng(seed)
        best_loss = float('inf')
        best_params: tuple[np.ndarray, ...] | None = None
        wait = 0
        history = {'train_loss': [], 'val_loss': []}

        y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)

        for _ in range(epochs):
            order = rng.permutation(len(x_train))
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]

                pred, (z1, a1) = self.forward(xb)
                error = pred - yb
                n = max(1, len(xb))
                d2 = (2.0 / n) * error
                dw2 = a1.T @ d2
                db2 = d2.sum(axis=0, keepdims=True)
                d1 = (d2 @ self.w2.T) * self.relu_grad(z1)
                dw1 = xb.T @ d1
                db1 = d1.sum(axis=0, keepdims=True)

                self.w2 -= learning_rate * dw2
                self.b2 -= learning_rate * db2
                self.w1 -= learning_rate * dw1
                self.b1 -= learning_rate * db1

            train_pred = self.predict(x_train)
            train_loss = float(np.mean((train_pred - y_train.ravel()) ** 2))
            val_loss = train_loss if x_val is None else float(np.mean((self.predict(x_val) - y_val.ravel()) ** 2))
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss + 1e-8 < best_loss:
                best_loss = val_loss
                best_params = (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_params is not None:
            self.w1, self.b1, self.w2, self.b2 = best_params
        return history

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> tuple['DenseNetwork', dict[str, Any]]:
        payload = np.load(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        model = cls(int(meta['input_size']), int(meta['hidden_size']))
        model.w1 = payload['w1']
        model.b1 = payload['b1']
        model.w2 = payload['w2']
        model.b2 = payload['b2']
        return model, meta
