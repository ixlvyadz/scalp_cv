import json
import numpy as np, pandas as pd
from pathlib import Path

R = Path("reports")
df = pd.read_csv(R / "split_final.csv")
z = np.load(sorted(R.glob("emb_*.npz"))[0])
E, Em = z["E"].astype(np.float32), z["Em"].astype(np.float32)
idx = {p: i for i, p in enumerate(df.path)}
classes = json.load(open("artifacts/labels.json"))

names = [f"{m}_seed{s}" for m in ("resnet50", "convnext_tiny", "mobilenetv3") for s in (1, 2, 3)]
P = [pd.read_csv(R / f"preds_test_{n}.csv") for n in names]
wrong = np.all([(p.pred != p.label).to_numpy() for p in P], axis=0)

tr = np.where(df.final == "train")[0]
for k in np.where(wrong)[0]:
    i = idx[P[0].path[k]]
    s = np.maximum(E[tr] @ E[i], Em[tr] @ E[i])
    order = np.argsort(-s)[:5]
    predicted = sorted({classes[p.pred[k]] for p in P})
    print(f"{P[0].path[k]}  (group {P[0].group[k]})  predicted as: {predicted}")
    print("   5 nearest training images:", [df.label[tr[j]] for j in order], s[order].round(2).tolist())