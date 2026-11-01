import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

df = pd.read_csv("reports/leakage.csv")
out = Path("reports/figures/leak_review_bands")
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

for lo, hi in [(7, 8), (9, 10), (11, 12), (13, 14)]:
    sub = df[df.hamming.between(lo, hi)]
    print(f"distance {lo}-{hi}: {len(sub)} pairs")
    if len(sub):
        sheet(list(sub.sample(min(24, len(sub)), random_state=0).iterrows()), f"dist_{lo}_{hi}")