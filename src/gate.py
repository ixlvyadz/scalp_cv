"""Optional "is this a scalp/hair photo?" check (trained by src/train_gate.py).

If artifacts/gate.onnx is missing, every photo is treated as a scalp photo, so the
rest of the project works exactly as before.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

ART = Path(__file__).resolve().parents[1] / "artifacts"


@lru_cache(maxsize=1)
def _load():
    cfg_path = ART / "gate_config.json"
    if not cfg_path.exists():
        return None, None
    cfg = json.load(open(cfg_path))
    sess = ort.InferenceSession(str(ART / cfg["model_file"]), providers=["CPUExecutionProvider"])
    return cfg, sess


def scalp_score(x):
    """x: preprocessed image (1, 3, 224, 224). Returns (score 0-1, is_scalp) or (None, True)."""
    cfg, sess = _load()
    if sess is None:
        return None, True
    z = sess.run(None, {"image": x})[0][0].astype(np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    s = float(p[cfg["scalp_index"]])
    return s, s >= cfg["threshold"]
