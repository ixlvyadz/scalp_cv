import numpy as np, pandas as pd, imagehash
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

DATA, OUT = Path("data"), Path("reports")
OUT.mkdir(exist_ok=True)
EXT = {".jpg", ".jpeg", ".png"}
THRESH = 14   # collect candidates up to this Hamming distance; fix the cut-off after visual review

TRANSFORMS = {
    "original": lambda im: im,
    "mirror":   ImageOps.mirror,
    "flip":     ImageOps.flip,
    "rot90":    lambda im: im.rotate(90, expand=True),
    "rot180":   lambda im: im.rotate(180),
    "rot270":   lambda im: im.rotate(270, expand=True),
}
NAMES = list(TRANSFORMS)

def phash_bits(im):
    return np.array(imagehash.phash(im).hash, dtype=bool).ravel()

# 1. Index every image
rows = [(s, p.parent.name, p.as_posix())
        for s in ["train", "val", "test"]
        for p in sorted((DATA / s).rglob("*")) if p.suffix.lower() in EXT]
df = pd.DataFrame(rows, columns=["split", "label", "path"])
print(df.groupby(["split", "label"]).size().unstack(0))          # class counts per split

# 2. Hash: train once, val/test in all six orientations
base = np.zeros((len(df), 64), dtype=bool)
var = {}
for i, r in tqdm(df.iterrows(), total=len(df), desc="hashing"):
    im = Image.open(r.path).convert("RGB")
    if r.split == "train":
        base[i] = phash_bits(im)
    else:
        var[i] = np.stack([phash_bits(t(im)) for t in TRANSFORMS.values()])
        base[i] = var[i][0]

# 3. Compare: closest reference image for each query image, over all orientations
def audit(query_split, ref_split):
    ref = df.index[df.split == ref_split].to_numpy()
    R = base[ref]
    out = []
    for qi in tqdm(df.index[df.split == query_split], desc=f"{query_split} vs {ref_split}"):
        d = (var[qi][:, None, :] != R[None, :, :]).sum(axis=2)    # (6 orientations, n_ref)
        t, j = np.unravel_index(d.argmin(), d.shape)
        if d[t, j] <= THRESH:
            m = ref[j]
            out.append((query_split, df.path[qi], df.label[qi], ref_split,
                        df.path[m], df.label[m], int(d[t, j]), NAMES[t]))
    return pd.DataFrame(out, columns=["split", "image", "label", "ref_split",
                                      "match_image", "match_label", "hamming", "transform"])

res = pd.concat([audit("val", "train"), audit("test", "train"), audit("test", "val")],
                ignore_index=True)
res["cross_class"] = res.label != res.match_label
res.to_csv(OUT / "leakage.csv", index=False)

print("\nflagged, by comparison:\n", res.groupby(["split", "ref_split"]).size())
print("\nflagged, by orientation that matched:\n", res.groupby("transform").size())
print("\nflagged, by Hamming distance:\n", res.groupby(["split", "hamming"]).size().unstack(0))
print("\ncross-class matches:", int(res.cross_class.sum()))