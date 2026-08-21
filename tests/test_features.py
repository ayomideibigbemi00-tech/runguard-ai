import numpy as np
from app.services.data import _synthetic
from app.services.features import make_supervised

def test_supervised_shapes():
    df = _synthetic('hourly', 200)
    x, y, _, feat = make_supervised(df, 6)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 30 * 12
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert len(feat) > 0
