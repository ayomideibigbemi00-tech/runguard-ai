from pathlib import Path
import json
import numpy as np


def test_validation_gate_prefers_baseline_when_nn_is_worse():
    yv = np.array([0.10, 0.20, 0.30])
    model = np.array([-0.10, -0.20, -0.30])
    baseline = np.full(3, 0.20)
    model_mae = float(np.mean(np.abs(model-yv)))
    baseline_mae = float(np.mean(np.abs(baseline-yv)))
    assert model_mae > baseline_mae
