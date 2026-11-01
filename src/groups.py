import numpy as np, pandas as pd, imagehash
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

DATA, OUT = Path("data"), Path("reports")
EXT = {".jpg", ".jpeg", ".png"}
T = 8   # grouping threshold: try 6, 8 and 10

rows = [(s, p.parent.name, p.as_posix()) for s in ["train", "val", "test"]
        for p in sorted((DATA / s).rglob("*")) if p.suffix.lower() in EXT]
df = pd.DataFrame(rows, columns=["split", "label", "path"])

def bits(im):
    return np.array(imagehash.phash(im).hash, dtype=bool).ravel()

cache = OUT / "hashes_mirror.npz"
if cache.exists():
    z = np.load(cache); H, Hm = z["H"], z["Hm"]
else:
    H = np.zeros((len(df), 64), dtype=bool); Hm = np.zeros_like(H)
    for i, p in enumerate(tqdm(df.path, desc="hashing")):
        im = Image.open(p).convert("RGB")
        H[i], Hm[i] = bits(im), bits(ImageOps.mirror(im))
    np.savez(cache, H=H, Hm=Hm)

S, Sm = np.where(H, 1, -1).astype(np.float32), np.where(Hm, 1, -1).astype(np.float32)
ii, jj = [], []
for a in tqdm(range(0, len(S), 1000), desc="comparing"):
    D = np.minimum((64 - S[a:a+1000] @ S.T) / 2, (64 - Sm[a:a+1000] @ S.T) / 2)
    r, c = np.where(D <= T)
    keep = (r + a) < c
    ii.append(r[keep] + a); jj.append(c[keep])
ii, jj = np.concatenate(ii), np.concatenate(jj)
g = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(df), len(df)))
n, df["group"] = connected_components(g, directed=False)

size = df.group.map(df.group.value_counts())
print(f"\nT={T}: {n} groups, {(size == 1).sum()} images with no copy, largest group {size.max()} images")
print("group sizes, top 5:", sorted(df.group.value_counts().head(5).tolist(), reverse=True))
spl = df.groupby("group").split.nunique()
print("groups spanning more than one split:", int((spl > 1).sum()))
print("groups with more than one label:", int((df.groupby("group").label.nunique() > 1).sum()))
has_train = df.group.isin(df[df.split == "train"].group)
for s in ["val", "test"]:
    print(f"{s}: {has_train[df.split == s].mean() * 100:.1f}% of images share a group with a train image")
df.to_csv(OUT / f"groups_T{T}.csv", index=False)