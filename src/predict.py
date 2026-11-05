import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from src.gate import scalp_score

ART = Path(__file__).resolve().parents[1] / "artifacts"


@lru_cache(maxsize=1)
def get_cfg():
    return json.load(open(ART / "preprocess_config.json"))


@lru_cache(maxsize=1)
def get_session():
    return ort.InferenceSession(str(ART / get_cfg()["model_file"]), providers=["CPUExecutionProvider"])


def preprocess(img, cfg=None):
    """Same steps as the evaluation transform used in training: resize, center crop, normalize."""
    cfg = cfg or get_cfg()
    size, resize = cfg["img_size"], cfg["resize"]
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = resize, int(resize * h / w)
    else:
        nw, nh = int(resize * w / h), resize
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(cfg["mean"], dtype=np.float32)) / np.array(cfg["std"], dtype=np.float32)
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def predict(img, top_k=3):
    cfg = get_cfg()
    x = preprocess(ImageOps.exif_transpose(img), cfg)         # phone photos carry a rotation tag
    gate, is_scalp = scalp_score(x)                           # gate uses the same 224 crop
    logits = get_session().run(None, {"image": x})[0][0].astype(np.float64)
    z = logits / cfg["temperature"]
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)[:top_k]
    return {"top": [(cfg["classes"][i], float(p[i])) for i in order],
            "confidence": float(p.max()),
            "uncertain": bool(p.max() < cfg["reject_threshold"]),
            "is_scalp": bool(is_scalp),                          # False: not a scalp/hair photo
            "scalp_score": gate}