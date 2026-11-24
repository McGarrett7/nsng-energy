"""Fit detector agents on training groups and emit score and confidence beliefs."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import torch
from torch import nn

from .data import Sequence

NAMES = (
    "IForest_0.05", "IForest_0.1", "OCSVM_0.05", "OCSVM_0.1",
    "LOF", "LSTM Autoencoder", "Adaptive Z Score",
)


@dataclass
class Beliefs:
    scores: np.ndarray
    confidence: np.ndarray


def rolling(power: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    history = pd.Series(power).shift(1).rolling(window, min_periods=1)
    mean = history.mean().fillna(power[0]).to_numpy()
    std = history.std(ddof=0).fillna(0).to_numpy()
    return mean, std


def features(power: np.ndarray, window: int) -> np.ndarray:
    mean, std = rolling(power, window)
    delta = np.diff(power, prepend=power[0])
    active_fraction = pd.Series(power > 10).rolling(window, min_periods=1).mean()
    return np.column_stack((power, delta, mean, std, power - mean, active_fraction))


def windows(values: np.ndarray, length: int) -> np.ndarray:
    padded = np.pad(values, (length - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, length)


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (encoding, _) = self.encoder(inputs)
        repeated = encoding[-1][:, None, :].expand(-1, inputs.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class Agents:
    def __init__(self, seed: int, device: torch.device, window: int, epochs: int):
        self.seed = seed
        self.device = device
        self.window = window
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.models = (
            IsolationForest(contamination=0.05, random_state=seed),
            IsolationForest(contamination=0.1, random_state=seed),
            OneClassSVM(nu=0.05),
            OneClassSVM(nu=0.1),
            LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True),
        )
        self.lstm = LSTMAutoencoder().to(device)

    def _power_windows(self, power: np.ndarray) -> np.ndarray:
        values = (power - self.power_center) / self.power_scale
        return windows(values, self.window)

    def _reconstruction_error(self, inputs: np.ndarray) -> np.ndarray:
        values = torch.tensor(inputs[:, :, None].copy(), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            errors = [
                (self.lstm(batch) - batch).square().mean(dim=1).squeeze(-1).cpu().numpy()
                for batch in values.split(256)
            ]
        return np.concatenate(errors)

    def _raw_scores(self, power, statistical_features, inputs) -> np.ndarray:
        values = self.scaler.transform(statistical_features)
        scores = [-model.decision_function(values) for model in self.models]
        mean, std = statistical_features[:, 2], statistical_features[:, 3]
        z = np.abs(power - mean) / np.hypot(std, self.z_floor)
        return np.column_stack((*scores, self._reconstruction_error(inputs), z))

    def fit(self, sequences: Iterable[Sequence]) -> None:
        references = {}
        for sequence in sequences:
            if sequence.normal_power is not None:
                references.setdefault(sequence.normal_key, (sequence.normal_power, None))
            else:
                references[sequence.name] = (sequence.power, sequence.labels)
        clean = []
        for power, labels in references.values():
            mask = np.ones(len(power), dtype=bool)
            if labels is not None:
                # A row is clean only if its causal feature history is clean as well.
                mask = ~windows(labels, self.window + 1).any(axis=1)
            if mask.any():
                clean.append((power, mask))
        if not clean or sum(mask.sum() for _, mask in clean) < 21:
            raise ValueError("The training split requires at least 21 clean power windows.")
        normal_power = np.concatenate([power[mask] for power, mask in clean])
        self.power_center = float(normal_power.mean())
        self.power_scale = float(max(normal_power.std(), 1))
        self.z_floor = max(float(np.quantile(np.abs(normal_power), 0.95)) * 0.01, 1)
        train_features = np.concatenate([features(power, self.window)[mask] for power, mask in clean])
        scaled = self.scaler.fit_transform(train_features)
        for model in self.models:
            model.fit(scaled)

        inputs = np.concatenate([self._power_windows(power)[mask] for power, mask in clean])
        values = torch.tensor(inputs[:, :, None], dtype=torch.float32, device=self.device)
        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.002)
        self.lstm.train()
        for _ in range(self.epochs):
            for indices in torch.randperm(len(values), generator=generator).split(256):
                batch = values[indices.to(self.device)]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.lstm(batch), batch)
                loss.backward()
                optimizer.step()
        self.lstm.eval()

        raw = np.concatenate([
            self._raw_scores(power, features(power, self.window), self._power_windows(power))[mask]
            for power, mask in clean
        ])
        self.center = np.array([0, 0, 0, 0, 0, np.quantile(raw[:, 5], 0.95), 3.0])
        scale = np.quantile(np.abs(raw - self.center), 0.75, axis=0)
        self.scale = np.maximum(scale / 2, 1e-6)
        # Older power meters can report long runs of exact zero.
        self.scale[-1] = max(self.scale[-1], 0.5)

    def score(self, power: np.ndarray) -> Beliefs:
        raw = self._raw_scores(power, features(power, self.window), self._power_windows(power))
        scores = expit((raw - self.center) / self.scale).astype(np.float32)
        confidence = 0.5 + np.abs(scores - 0.5)
        return Beliefs(scores, confidence)
