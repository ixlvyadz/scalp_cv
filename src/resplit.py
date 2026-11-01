import numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA, OUT = Path("data"), Path("reports")
EXT = {".jpg", ".jpeg", ".png"}
T = 8          # images within this distance are treated as copies of each other
PURGE = 12     # val/test images still this close to a train image are dropped
SEED = 0
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}

rows = [(s, p.parent.name, p.as_posix()) for s in ["train", "val", "test"]
        for p in sorted((DATA / s).rglob("*")) if p.suffix.lower() in EXT]
df = pd.DataFrame(rows, columns=["orig_split", "label", "path"])
z = np.load(OUT / "hashes_mirror.npz")
H, Hm = z["H"], z["Hm"]
assert len(H) == len(df), "hash cache does not match the images on disk"
N = len(df)

# 1. all image pairs within distance 14 (mirror-aware)
S = np.where(H, 1, -1).astype(np.float32)
Sm = np.where(Hm, 1, -1).astype(np.float32)
R, C, D = [], [], []
for a in tqdm(range(0, N, 1000), desc="distances"):
    d = np.minimum((64 - S[a:a+1000] @ S.T) / 2, (64 - Sm[a:a+1000] @ S.T) / 2)
    r, c = np.where(d <= 14)
    keep = (r + a) < c
    R.append((r[keep] + a).astype(np.int32)); C.append(c[keep].astype(np.int32))
    D.append(d[r[keep], c[keep]].astype(np.int8))
R, C, D = np.concatenate(R), np.concatenate(C), np.concatenate(D)
print(len(R), "image pairs within distance 14")

# 2. leader clustering: no chaining, groups only gather images close to their first image
adj = [[] for _ in range(N)]
for r, c, d in zip(R, C, D):
    if d <= T:
        adj[r].append(c); adj[c].append(r)
group = np.full(N, -1)
g = 0
for i in np.random.default_rng(SEED).permutation(N):
    if group[i] < 0:
        group[i] = g
        for j in adj[i]:
            if group[j] < 0:
                group[j] = g
        g += 1
df["group"] = group
sizes = df.group.value_counts()
print(f"{g} groups, largest {sizes.iloc[0]}, singletons {(sizes == 1).sum()}")

# 3. set aside groups whose images carry different labels
nl = df.groupby("group").label.nunique()
df["conflict"] = df.group.isin(nl[nl > 1].index)
print(f"{(nl > 1).sum()} groups with conflicting labels ({df.conflict.sum()} images set aside)")

# 4. assign whole groups to splits, class by class
assign = {}
for label, sub in df[~df.conflict].groupby("label"):
    sz = sub.groupby("group").size().sample(frac=1, random_state=SEED)
    sz = sz.sort_values(ascending=False, kind="stable")
    need = {s: TARGET[s] * sz.sum() for s in TARGET}
    have = {s: 0 for s in TARGET}
    for gid, n in sz.items():
        s = max(TARGET, key=lambda k: need[k] - have[k])
        assign[gid] = s
        have[s] += n
df["split"] = df.group.map(assign).fillna("conflict")

# 5. drop the train-side partner of every close cross-split pair; val and test stay intact
sp = df.split.to_numpy()
drop = np.zeros(N, dtype=bool)
for r, c, d in zip(R, C, D):
    if d <= PURGE:
        pair = {sp[r], sp[c]}
        if pair == {"train", "val"} or pair == {"train", "test"}:
            drop[r if sp[r] == "train" else c] = True
        elif pair == {"val", "test"}:
            drop[r if sp[r] == "val" else c] = True
df["final"] = np.where(drop, "dropped", df.split)

print(f"\nPURGE={PURGE}: dropped {int(drop.sum())} images")
print(df.pivot_table(index="label", columns="final", values="path", aggfunc="count", fill_value=0))
print(df.final.value_counts())

# 6. leakage still visible at looser distances among the kept images
keep = df.final.isin(["train", "val", "test"]).to_numpy()
fin = df.final.to_numpy()
rem = np.full(N, 99)
for r, c, d in zip(R, C, D):
    if keep[r] and keep[c]:
        for x, y in ((r, c), (c, r)):
            if fin[y] == "train" and fin[x] in ("val", "test"):
                rem[x] = min(rem[x], d)
for s in ["val", "test"]:
    m = df.final == s
    print(s, "share with a train image within k after cleaning:",
          {k: f"{(rem[m] <= k).mean() * 100:.1f}%" for k in (8, 10, 12, 14)})
df.to_csv(OUT / "split.csv", index=False)