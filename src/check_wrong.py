import numpy as np, pandas as pd
from pathlib import Path

R = Path("reports")
df = pd.read_csv(R / "split_final.csv")
z = np.load(sorted(R.glob("emb_*.npz"))[0])
E, Em = z["E"].astype(np.float32), z["Em"].astype(np.float32)
assert len(E) == len(df)
idx = {p: i for i, p in enumerate(df.path)}

names = [f"{m}_seed{s}" for m in ("resnet50", "convnext_tiny", "mobilenetv3") for s in (1, 2, 3)]
P = [pd.read_csv(R / f"preds_test_{n}.csv") for n in names]
wrong = np.all([(p.pred != p.label).to_numpy() for p in P], axis=0)
paths = P[0].path[wrong].tolist()

conf = np.where(df.final == "conflict")[0]
print(len(conf), "images were set aside for conflicting labels\n")
for p in paths:
    i = idx[p]
    s = np.maximum(E[conf] @ E[i], Em[conf] @ E[i])
    j = s.argmax()
    print(f"{p}\n   closest set-aside image: {df.path[conf[j]]} (label {df.label[conf[j]]}), similarity {s[j]:.3f}")