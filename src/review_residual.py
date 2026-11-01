import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

OUT = Path("reports")
df = pd.read_csv(OUT / "split_p12.csv")
z = np.load(OUT / "hashes_mirror.npz")
S = np.where(z["H"], 1, -1).astype(np.float32)
Sm = np.where(z["Hm"], 1, -1).astype(np.float32)
tr = np.where(df.final == "train")[0]
St = S[tr].T

pairs = []
for i in np.where(df.final.isin(["val", "test"]))[0]:
    d = np.minimum((64 - S[i] @ St) / 2, (64 - Sm[i] @ St) / 2)
    j = d.argmin()
    pairs.append((i, tr[j], int(d[j])))
P = pd.DataFrame(pairs, columns=["i", "j", "d"])
print(P.d.value_counts().sort_index().head(10))     # nearest-train distance for every kept val/test image

sheetdir = OUT / "figures" / "leak_residual"
sheetdir.mkdir(parents=True, exist_ok=True)
for dist in sorted(P.d.unique()):
    if dist < 12 or dist > 14:
        continue
    sub = P[P.d == dist].sample(min(24, (P.d == dist).sum()), random_state=0)
    rows = list(sub.itertuples())
    for k in range(0, len(rows), 8):
        chunk = rows[k:k + 8]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 3 * len(chunk)), squeeze=False)
        for r, (a, b) in zip(chunk, axes):
            a.imshow(Image.open(df.path[r.i])); a.set_title(f"{df.final[r.i]}, d={r.d}", fontsize=7)
            b.imshow(Image.open(df.path[r.j])); b.set_title("nearest train image", fontsize=7)
            a.axis("off"); b.axis("off")
        plt.tight_layout()
        plt.savefig(sheetdir / f"d{dist}_{k // 8}.png", dpi=150)
        plt.close()
print("sheets in", sheetdir)