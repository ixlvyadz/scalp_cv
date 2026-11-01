import numpy as np, pandas as pd
from pathlib import Path

OUT, DATA = Path("reports"), Path("data")
EXT = {".jpg", ".jpeg", ".png"}
z = np.load(sorted(OUT.glob("emb_*.npz"))[0])
E, Em = z["E"].astype(np.float32), z["Em"].astype(np.float32)

rows = [(s, p.parent.name, p.as_posix()) for s in ["train", "val", "test"]
        for p in sorted((DATA / s).rglob("*")) if p.suffix.lower() in EXT]
df = pd.DataFrame(rows, columns=["split", "label", "path"])
idx = {p: i for i, p in enumerate(df.path)}
assert len(E) == len(df)

def sim(a, b):
    return max(E[a] @ E[b], Em[a] @ E[b])

lk = pd.read_csv(OUT / "leakage.csv")
known = lk[lk.hamming <= 4]                      # confirmed same-photo pairs on your sheets
s_known = [sim(idx[a], idx[b]) for a, b in zip(known.image, known.match_image)]

rng = np.random.default_rng(0)
same, diff = [], []
for _ in range(20000):
    a, b = rng.integers(0, len(df), 2)
    if a != b:
        (same if df.label[a] == df.label[b] else diff).append(sim(a, b))

q = [1, 5, 25, 50, 75, 95, 99, 99.9]
tab = pd.DataFrame({"known copies (pHash<=4)": np.percentile(s_known, q),
                    "random, same class": np.percentile(same, q),
                    "random, different class": np.percentile(diff, q)}, index=[f"p{x}" for x in q])
print(len(s_known), "known copy pairs")
print(tab.round(3).to_string())