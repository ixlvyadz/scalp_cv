import numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

OUT, DATA = Path("reports"), Path("data")
EXT = {".jpg", ".jpeg", ".png"}
SIM = 0.90      # images at least this similar count as copies; set from embed_calibrate
GAP = 0.03      # afterwards drop train/val images within SIM - GAP of a kept val/test image
SEED = 0
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}

rows = [(s, p.parent.name, p.as_posix()) for s in ["train", "val", "test"]
        for p in sorted((DATA / s).rglob("*")) if p.suffix.lower() in EXT]
df = pd.DataFrame(rows, columns=["orig_split", "label", "path"])
z = np.load(sorted(OUT.glob("emb_*.npz"))[0])
E, Em = z["E"].astype(np.float32), z["Em"].astype(np.float32)
assert len(E) == len(df), "embeddings do not match the images on disk"
N = len(df)

def sims(i, pool=None):
    a = E if pool is None else E[pool]
    return np.maximum(a @ E[i], a @ Em[i])        # mirror-aware cosine similarity

# 1. leader clustering: each group gathers images close to its first image, so no chaining
group = np.full(N, -1)
g = 0
for i in tqdm(np.random.default_rng(SEED).permutation(N), desc="grouping"):
    if group[i] < 0:
        members = np.where((sims(i) >= SIM) & (group < 0))[0]
        group[members] = g
        group[i] = g
        g += 1
df["group"] = group
sizes = df.group.value_counts()
print(f"\nSIM={SIM}: {g} groups (about {g} independent images), largest {sizes.iloc[0]}, "
      f"singletons {(sizes == 1).sum()}")
print("group size quantiles (50/90/99%):", sizes.quantile([.5, .9, .99]).tolist())

# 2. set aside groups with conflicting labels
nl = df.groupby("group").label.nunique()
df["conflict"] = df.group.isin(nl[nl > 1].index)
print(f"{(nl > 1).sum()} groups with conflicting labels ({df.conflict.sum()} images set aside)")

# 3. assign whole groups to splits, class by class
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
# 4. closeness of train images to val/test images (mirror-aware)
sp = df.split.to_numpy()
lab = df.label.to_numpy()
tr = np.where(sp == "train")[0]
va = np.where(sp == "val")[0]
te = np.where(sp == "test")[0]
vt = np.concatenate([va, te])

def maxsim(rows, cols):
    out = np.zeros(len(rows), dtype=np.float32)
    for a in range(0, len(rows), 1000):
        r = rows[a:a + 1000]
        m = np.maximum(E[r] @ E[cols].T, Em[r] @ E[cols].T)
        out[a:a + 1000] = m.max(axis=1)
    return out

tr_max = maxsim(tr, vt)      # for each train image: its closest val/test image
va_max = maxsim(va, te)      # for each val image: its closest test image

print("\nSweep: images kept at each purge threshold")
classes = sorted(set(lab))
for t in [0.87, 0.90, 0.92, 0.94, 0.96, 0.98, 1.01]:
    cnt = pd.Series(lab[tr][tr_max < t]).value_counts().reindex(classes, fill_value=0)
    print(f"  t={t}: train {int(cnt.sum())}, smallest class {int(cnt.min())}, val kept {int((va_max < t).sum())}")

# 5. apply the chosen threshold
PURGE_T = 0.92     # set this after reading the sweep
drop = np.zeros(N, dtype=bool)
drop[tr[tr_max >= PURGE_T]] = True
drop[va[va_max >= PURGE_T]] = True
df["final"] = np.where(drop, "dropped", df.split)
print(f"\nPURGE_T={PURGE_T}")
print(df.pivot_table(index="label", columns="final", values="path", aggfunc="count", fill_value=0))
print(df.final.value_counts())

# 6. record how close each kept val/test image is to its nearest kept train image
fin = df.final.to_numpy()
tr_kept = np.where(fin == "train")[0]
df["nn_train_sim"] = np.nan
for s in ["val", "test"]:
    rows_ = np.where(fin == s)[0]
    df.loc[rows_, "nn_train_sim"] = maxsim(rows_, tr_kept)
    m = df.loc[rows_, "nn_train_sim"]
    print(f"{s}: nearest-train similarity  median {m.median():.3f}  p95 {m.quantile(.95):.3f}  max {m.max():.3f}")
    print("   share by band:", pd.cut(m, [0, .85, .90, .95, 1.01]).value_counts(normalize=True).sort_index().round(3).to_dict())
df.to_csv(OUT / "split_emb.csv", index=False)