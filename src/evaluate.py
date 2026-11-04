import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)

from src.dataset import get_loaders


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def bootstrap_ci(y, p, groups, n=1000, seed=0):
    """95% interval for macro-F1, resampling whole groups (pass groups=arange for the naive version)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    members = [np.where(groups == g)[0] for g in ids]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        sel = np.concatenate([members[k] for k in pick])
        vals.append(f1_score(y[sel], p[sel], average="macro", zero_division=0))
    return np.percentile(vals, [2.5, 97.5])


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets, loaders = get_loaders(num_workers=0)
    ds = sets[a.split]
    classes = ds.classes

    model = timm.create_model(a.model, pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.to(device)

    logits, y = predict(model, loaders[a.split], device)
    p = logits.argmax(1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    groups = ds.groups

    macro = f1_score(y, p, average="macro", zero_division=0)
    acc = accuracy_score(y, p)
    bal = balanced_accuracy_score(y, p)
    lo, hi = bootstrap_ci(y, p, groups)
    nlo, nhi = bootstrap_ci(y, p, np.arange(len(y)))

    print(f"\n{a.name} on {a.split}: {len(y)} images, {len(np.unique(groups))} groups")
    print(f"macro-F1 {macro:.4f}   accuracy {acc:.4f}   balanced accuracy {bal:.4f}")
    print(f"95% interval, group bootstrap : [{lo:.4f}, {hi:.4f}]")
    print(f"95% interval, naive (images)  : [{nlo:.4f}, {nhi:.4f}]")

    per_class = pd.DataFrame({
        "class": classes,
        "recall": recall_score(y, p, average=None, labels=range(len(classes)), zero_division=0),
        "images": np.bincount(y, minlength=len(classes)),
        "groups": [len(np.unique(groups[y == k])) for k in range(len(classes))],
    })
    print("\nper-class recall:\n", per_class.round(3).to_string(index=False))

    # breakdown by closeness to the nearest training image
    sim = ds.nn_sim
    bands = pd.qcut(pd.Series(sim), 3, labels=["lowest third", "middle third", "highest third"])
    rows = []
    for b in bands.cat.categories:
        m = (bands == b).to_numpy()
        rows.append({"similarity to train": b, "images": int(m.sum()),
                     "sim range": f"{sim[m].min():.3f}-{sim[m].max():.3f}",
                     "accuracy": accuracy_score(y[m], p[m]),
                     "macro-F1": f1_score(y[m], p[m], average="macro", zero_division=0)})
    band_tab = pd.DataFrame(rows)
    print("\nby similarity to the nearest training image:\n", band_tab.round(4).to_string(index=False))

    # save everything
    out = Path("reports"); (out / "figures").mkdir(parents=True, exist_ok=True)
    tag = f"{a.split}_{a.name}"
    np.save(out / f"logits_{tag}.npy", logits)
    pd.DataFrame({"path": [s[0] for s in ds.samples], "label": y, "pred": p,
                  "conf": probs.max(1), "group": groups, "nn_sim": sim}
                 ).to_csv(out / f"preds_{tag}.csv", index=False)
    per_class.to_csv(out / f"per_class_{tag}.csv", index=False)
    band_tab.to_csv(out / f"bands_{tag}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix(y, p), display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout(); plt.savefig(out / "figures" / f"cm_{tag}.png", dpi=200); plt.close()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"scalp-cv-{a.split}")
    with mlflow.start_run(run_name=a.name):
        mlflow.log_metrics({f"{a.split}_macro_f1": macro, f"{a.split}_acc": acc,
                            f"{a.split}_bal_acc": bal, "ci_group_low": lo, "ci_group_high": hi,
                            "ci_naive_low": nlo, "ci_naive_high": nhi})
        mlflow.log_artifact(str(out / "figures" / f"cm_{tag}.png"))
        mlflow.log_artifact(str(out / f"per_class_{tag}.csv"))
        mlflow.log_artifact(str(out / f"bands_{tag}.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    main(ap.parse_args())