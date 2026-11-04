import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import timm
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import get_loaders

MODEL = "mobilenetv3_large_100"
CKPT = "artifacts/checkpoints/mobilenetv3_large_seed1.pt"
LOGITS = "reports/logits_test_mobilenetv3_seed1.npy"
A, R = Path("artifacts"), Path("reports")
FP32, INT8 = A / "model_fp32.onnx", A / "model_int8.onnx"

classes = json.load(open(A / "labels.json"))
sets, loaders = get_loaders(num_workers=0, batch_size=32)
y = np.array(sets["test"].targets)

# 1. export to ONNX
model = timm.create_model(MODEL, pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
kw = dict(input_names=["image"], output_names=["logits"],
          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
dummy = torch.randn(1, 3, 224, 224)
try:
    torch.onnx.export(model, dummy, str(FP32), dynamo=False, **kw)
except TypeError:                                   # older PyTorch without the dynamo argument
    torch.onnx.export(model, dummy, str(FP32), **kw)
print("exported", FP32)


# 2. static INT8 quantization, calibrated on validation images
class Reader(CalibrationDataReader):
    def __init__(self, arrays):
        self.it = iter([{"image": a} for a in arrays])

    def get_next(self):
        return next(self.it, None)


val = sets["val"]
idx = np.random.default_rng(0).choice(len(val), 200, replace=False)
try:
    quantize_static(str(FP32), str(INT8), Reader([val[i][0].numpy()[None] for i in idx]),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    print("quantized", INT8)
except Exception as e:                              # keep going with FP32 only
    print("INT8 quantization failed:", repr(e))


# 3. measure
def run_onnx(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.concatenate([sess.run(None, {"image": x.numpy()})[0] for x, _ in loaders["test"]])
    return out, sess


def median_ms(fn, x, n=100, warm=10):
    for _ in range(warm):
        fn(x)
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, yy, bins=15):
    conf, ok = p.max(1), p.argmax(1) == yy
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return tot


cal = json.load(open(A / "calibration.json"))
ref = np.load(LOGITS)                               # logits saved by evaluate.py (PyTorch)
x1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
rows = []
with torch.no_grad():
    rows.append({"model": "PyTorch FP32 (CPU)", "size MB": np.nan,
                 "latency ms": median_ms(lambda x: model(torch.from_numpy(x)), x1)})
results = {}
for label, path in [("ONNX FP32", FP32), ("ONNX INT8", INT8)]:
    if not path.exists():
        continue
    lg, sess = run_onnx(path)
    results[label] = lg
    pred = lg.argmax(1)
    rows.append({"model": label, "size MB": path.stat().st_size / 1e6,
                 "latency ms": median_ms(lambda x: sess.run(None, {"image": x}), x1),
                 "test macro-F1": f1_score(y, pred, average="macro"),
                 "test accuracy": accuracy_score(y, pred),
                 "agrees with PyTorch": float((pred == ref.argmax(1)).mean()),
                 "ECE with temperature": ece(softmax(lg / cal["temperature"]), y)})
tab = pd.DataFrame(rows)
print("\n", tab.round(4).to_string(index=False))
tab.to_csv(R / "export_benchmark.csv", index=False)
print("max |FP32 ONNX - PyTorch| logit difference:", float(np.abs(results["ONNX FP32"] - ref).max()))

# 4. choose the deployed file: INT8 only if it keeps accuracy and is faster
chosen = "model_fp32.onnx"
if "ONNX INT8" in results:
    f32, i8 = tab[tab.model == "ONNX FP32"].iloc[0], tab[tab.model == "ONNX INT8"].iloc[0]
    if f32["test macro-F1"] - i8["test macro-F1"] <= 0.01 and i8["latency ms"] < f32["latency ms"]:
        chosen = "model_int8.onnx"
print("\ndeployed model file:", chosen)

cfg = {"classes": classes, "img_size": 224, "resize": 255, "mean": [0.485, 0.456, 0.406],
       "std": [0.229, 0.224, 0.225], "temperature": cal["temperature"],
       "reject_threshold": cal["reject_threshold"], "model_file": chosen}
json.dump(cfg, open(A / "preprocess_config.json", "w"), indent=2)

# 5. check that src/predict.py preprocesses exactly like the training pipeline
from src.predict import preprocess
test = sets["test"]
diff = max(float(np.abs(preprocess(Image.open(test.samples[i][0]), cfg)[0] - test[i][0].numpy()).max())
           for i in range(50))
print(f"max preprocessing difference over 50 test images: {diff:.2e}  (should be about 1e-6 or less)")