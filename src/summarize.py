import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

R = Path("reports")
MODELS = {"ResNet50": "resnet50", "ConvNeXt-Tiny": "convnext_tiny", "MobileNetV3-Large": "mobilenetv3"}
SEEDS = [1, 2, 3]
classes = json.load(open("artifacts/labels.json"))

P = {m: [pd.read_csv(R / f"preds_test_{tag}_seed{s}.csv") for s in SEEDS] for m, tag in MODELS.items()}
base = P["ResNet50"][0]
for lst in P.values():
    for d in lst:
        assert (d.path == base.path).all(), "test images are in a different order"
y = base.label.to_numpy()
groups = base.group.to_numpy()
sim = base.nn_sim.to_numpy()
band = pd.qcut(pd.Series(sim), 3, labels=["lowest", "middle", "highest"]).to_numpy().astype(str)
low = band == "lowest"
preds = {m: [d.pred.to_numpy() for d in lst] for m, lst in P.items()}


def f1(yy, pp):
    return f1_score(yy, pp, average="macro", zero_division=0)


def ms(v):
    return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"


# 1. headline table, mean ± sd over seeds
rows = []
for m in MODELS:
    rows.append({"model": m,
                 "macro-F1 (all test)": ms([f1(y, p) for p in preds[m]]),
                 "accuracy (all test)": ms([accuracy_score(y, p) for p in preds[m]]),
                 "accuracy, lowest-similarity third": ms([accuracy_score(y[low], p[low]) for p in preds[m]]),
                 "accuracy, other two thirds": ms([accuracy_score(y[~low], p[~low]) for p in preds[m]])})
t1 = pd.DataFrame(rows)
print("\n", t1.to_string(index=False))
t1.to_csv(R / "summary_models.csv", index=False)

# 2. per-class recall, mean over seeds
rec = pd.DataFrame({m: [np.mean([(p[y == k] == k).mean() for p in preds[m]])
                        for k in range(len(classes))] for m in MODELS}, index=classes)
print("\nper-class recall (mean over seeds):\n", rec.round(3))
rec.to_csv(R / "summary_per_class.csv")

# 3. class mix of the similarity bands (is the lowest third just the hard classes?)
mix = pd.crosstab(pd.Series(np.array(classes)[y], name="class"), pd.Series(band, name="band"))
print("\nclass mix by similarity band (images):\n", mix[["lowest", "middle", "highest"]])


# 4. paired group bootstrap on the seed-averaged macro-F1 difference
def boot_diff(a, b, mask, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups[mask])
    members = [np.where(mask & (groups == g))[0] for g in ids]

    def avg(m, sel):
        return np.mean([f1(y[sel], p[sel]) for p in preds[m]])

    full = np.where(mask)[0]
    point = avg(a, full) - avg(b, full)
    out = []
    for _ in range(n):
        sel = np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])
        out.append(avg(a, sel) - avg(b, sel))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi


print("\npaired group bootstrap, difference in macro-F1 (A minus B):")
for a, b in [("ConvNeXt-Tiny", "MobileNetV3-Large"), ("MobileNetV3-Large", "ResNet50"),
             ("ConvNeXt-Tiny", "ResNet50")]:
    for name, mask in [("all test", np.ones(len(y), bool)), ("lowest third", low)]:
        d, lo, hi = boot_diff(a, b, mask)
        print(f"  {a} - {b}, {name}: {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

# 5. images that every one of the 9 models gets wrong
wrong = np.all([p != y for lst in preds.values() for p in lst], axis=0)
print(f"\nimages wrong in all 9 runs: {wrong.sum()} (in {len(np.unique(groups[wrong]))} groups)")
print(pd.Series(np.array(classes)[y[wrong]]).value_counts().to_string())
all_idx = np.where(wrong)[0]
if len(all_idx):
    pick = pd.Series(all_idx).groupby(groups[all_idx]).first().to_numpy()[:16]
    ref = preds["ConvNeXt-Tiny"][0]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(Image.open(base.path[i]))
        ax.set_title(f"true: {classes[y[i]]}\npredicted: {classes[ref[i]]}", fontsize=8)
    plt.tight_layout()
    plt.savefig(R / "figures" / "always_wrong.png", dpi=150)
    print("saved reports/figures/always_wrong.png")

    # 6. accuracy by band, with and without the two hardest classes
hard = np.isin(np.array(classes)[y], ["Male Pattern Baldness", "Psoriasis"])
for name, m in [("all classes", np.ones(len(y), bool)),
                ("without Male Pattern Baldness and Psoriasis", ~hard)]:
    print(f"\n{name}: accuracy by band (mean over seeds)")
    for mod in MODELS:
        res = {b: round(float(np.mean([accuracy_score(y[m & (band == b)], p[m & (band == b)])
                                       for p in preds[mod]])), 4) for b in ["lowest", "middle", "highest"]}
        print(f"  {mod:18}", res)

        # 7. sensitivity: results without the one photo that every run gets wrong (secondary result only)
keep = ~wrong
psor = classes.index("Psoriasis")
for m in MODELS:
    f = [f1(y[keep], p[keep]) for p in preds[m]]
    r = [(p[keep & (y == psor)] == psor).mean() for p in preds[m]]
    print(f"{m:18} macro-F1 without the 10 images {ms(f)}   Psoriasis recall {ms(r)}")