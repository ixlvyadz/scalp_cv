import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

df = pd.read_csv("reports/leakage.csv")
out = Path("reports/figures/leak_review")
out.mkdir(parents=True, exist_ok=True)

def sheet(rows, name, per_sheet=8):
    for k in range(0, len(rows), per_sheet):
        chunk = rows[k:k + per_sheet]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 3 * len(chunk)), squeeze=False)
        for (_, r), (a, b) in zip(chunk, axes):
            a.imshow(Image.open(r.image))
            a.set_title(f"{r.split} vs {r.ref_split}, d={r.hamming}", fontsize=7)
            b.imshow(Image.open(r.match_image))
            b.set_title(f"match ({r['transform']})", fontsize=7)
            a.axis("off"); b.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"{name}_{k // per_sheet}.png", dpi=150)
        plt.close()

def pick(sub, n=None):
    sub = sub if n is None or len(sub) <= n else sub.sample(n, random_state=0)
    return list(sub.iterrows())

sheet(pick(df[df.hamming <= 2]), "dist_0_2")                          # all of them
sheet(pick(df[df.hamming.between(3, 4)], 30), "dist_3_4")             # sample
sheet(pick(df[df.hamming.between(5, 6)], 30), "dist_5_6")             # sample
sheet(pick(df[df["transform"] != "original"]), "mirrored_rotated")       # all of them
sheet(pick(df[df.ref_split == "val"]), "test_vs_val")                 # all of them
sheet(pick(df[df.cross_class]), "cross_class")                        # all of them
print("sheets saved in", out)