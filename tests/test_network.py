import numpy as np
from app.model.network import DenseNetwork

def test_network_learns_simple_relationship():
    rng = np.random.default_rng(7)
    x = np.linspace(-1, 1, 120).reshape(-1, 1)
    y = 2*x.ravel() + 0.25
    order = rng.permutation(len(x))
    train, test = order[:90], order[90:]
    model = DenseNetwork(1, hidden_size=12, seed=1)
    model.fit(x[train], y[train], x[test], y[test], epochs=150, learning_rate=0.01, patience=30)
    pred = model.predict(x[test])
    assert np.mean(np.abs(pred - y[test])) < 0.3
