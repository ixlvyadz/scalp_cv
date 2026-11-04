import numpy as np, pandas as pd
from pathlib import Path

R = Path("reports")
df = pd.read_csv(R / "split_final.csv")
z = np.load(sorted(R.glob("emb_*.npz"))[0])
E, Em = z["E"].astype(np.float32), z["Em"].astype(np.float32)
idx = {p: i for i, p in enumerate(df.path)}

names = [f"{m}_seed{s}" for m in ("resnet50", "convnext_tiny", "mobilenetv3") for s in (1, 2, 3)]
P = [pd.read_csv(R / f"preds_test_{n}.csv") for n in names]
wrong = np.all([(p.pred != p.label).to_numpy() for p in P], axis=0)

paths = P[0].path[wrong].tolist()
ii = [idx[p] for p in paths]
S = np.maximum(E[ii] @ E[ii].T, Em[ii] @ E[ii].T)
labels = [f"{Path(p).stem[-4:]}(g{g})" for p, g in zip(paths, P[0].group[wrong])]
print(pd.DataFrame(S.round(2), index=labels, columns=labels).to_string())