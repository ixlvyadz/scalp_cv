"""Train a "is this a scalp/hair photo?" gate that runs before the disease classifier.

Why: the 10-class classifier always picks one of its 10 conditions, even for a photo of
a laptop (it has no "none of these" option). The gate rejects such photos first.

Positives: every image in reports/split_final.csv (the same train/val/test split as the
classifier, so the gate's test set is the classifier's test set).
Negatives: Caltech-101 everyday objects (laptops, chairs, cameras, animals, ...), without
its face classes, plus any photos you put in data/not_scalp/ (recommended: 100-300 phone
photos of desks, screens, rooms, hands, pets, food...).

The threshold is chosen on validation images so that at most 1% of real scalp photos are
wrongly rejected (change with --max-false-reject). Test results are printed and saved.

Run from the repo root:
    python -m src.train_gate                      # full run (downloads Caltech-101 once)
    python -m src.train_gate --smoke             # 1 quick epoch to check everything works
Outputs:
    artifacts/gate.onnx, artifacts/gate_config.json, reports/gate_results.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataset import MEAN, STD, get_transforms

A, R = Path("artifacts"), Path("reports")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDE_CALTECH = {"Faces", "Faces_easy", "BACKGROUND_Google"}  # faces show hair; background is noise


class Paths(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


def scalp_items(csv):
    df = pd.read_csv(csv)
    return {s: [(p, 1) for p in df[df.final == s].path] for s in ["train", "val", "test"]}


def negative_paths(use_caltech, neg_dir, root):
    paths = []
    if use_caltech:
        from torchvision.datasets import Caltech101
        Caltech101(root=root, download=True)  # downloads once into data/
        base = Path(root) / "caltech101" / "101_ObjectCategories"
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name not in EXCLUDE_CALTECH:
                paths += [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    nd = Path(neg_dir)
    if nd.exists():
        paths += [p for p in sorted(nd.rglob("*")) if p.suffix.lower() in IMG_EXT]
    return [str(p) for p in paths]


def split_negatives(paths, seed):
    rng = random.Random(seed)
    paths = paths[:]
    rng.shuffle(paths)
    n = len(paths)
    a, b = int(0.7 * n), int(0.85 * n)
    return {"train": [(p, 0) for p in paths[:a]], "val": [(p, 0) for p in paths[a:b]],
            "test": [(p, 0) for p in paths[b:]]}


@torch.no_grad()
def scores(model, loader, device):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = scalp_items(a.csv)
    neg = split_negatives(negative_paths(not a.no_caltech, a.neg_dir, a.data_root), a.seed)
    if not neg["train"]:
        raise SystemExit("No negative images found. Allow Caltech-101 or add photos to data/not_scalp/.")
    if a.smoke:
        pos = {k: v[:64] for k, v in pos.items()}
        neg = {k: v[:64] for k, v in neg.items()}
    print({s: {"scalp": len(pos[s]), "not_scalp": len(neg[s])} for s in pos})

    train_tf, eval_tf = get_transforms(224)
    # Balance classes in each training batch.
    train_items = pos["train"] + neg["train"]
    w = np.array([1 / len(pos["train"]) if y else 1 / len(neg["train"]) for _, y in train_items])
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w), num_samples=len(train_items))
    tl = DataLoader(Paths(train_items, train_tf), batch_size=a.batch_size, sampler=sampler, num_workers=a.workers)
    ld = {s: DataLoader(Paths(pos[s] + neg[s], eval_tf), batch_size=64, num_workers=a.workers)
          for s in ["val", "test"]}

    model = timm.create_model(a.model, pretrained=not a.no_pretrained, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    epochs = 1 if a.smoke else a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        s, y = scores(model, ld["val"], device)
        bal = 0.5 * ((s[y == 1] >= 0.5).mean() + (s[y == 0] < 0.5).mean())
        print(f"epoch {ep}: val balanced accuracy {bal:.4f}")
        if bal > best:
            best, best_state = bal, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    # Threshold: reject at most `max_false_reject` of real scalp photos on validation.
    s, y = scores(model, ld["val"], "cpu")
    thr = float(np.quantile(s[y == 1], a.max_false_reject))
    st, yt = scores(model, ld["test"], "cpu")
    test_paths = [p for p, _ in pos["test"] + neg["test"]]
    res = {
        "threshold": thr,
        "val_scalp_pass_rate": float((s[y == 1] >= thr).mean()),
        "val_not_scalp_reject_rate": float((s[y == 0] < thr).mean()),
        "test_scalp_pass_rate": float((st[yt == 1] >= thr).mean()),
        "test_not_scalp_reject_rate": float((st[yt == 0] < thr).mean()),
        "test_counts": {"scalp": int((yt == 1).sum()), "not_scalp": int((yt == 0).sum())},
    }
    # Which scalp classes lose the most photos to the gate?
    df = pd.read_csv(a.csv).set_index("path")
    rejected = [p for p, sc, yy in zip(test_paths, st, yt) if yy == 1 and sc < thr]
    res["test_scalp_rejected_by_class"] = df.loc[rejected].label.value_counts().to_dict() if rejected else {}
    res["test_scalp_rejected_files"] = rejected
    print(json.dumps({k: v for k, v in res.items() if k != "test_scalp_rejected_files"}, indent=2))

    A.mkdir(exist_ok=True); R.mkdir(exist_ok=True)
    kw = dict(input_names=["image"], output_names=["logits"],
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    dummy = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(model, dummy, str(A / "gate.onnx"), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(A / "gate.onnx"), **kw)
    cfg = {"model_file": "gate.onnx", "img_size": 224, "resize": 255, "mean": list(MEAN), "std": list(STD),
           "scalp_index": 1, "threshold": thr, "backbone": a.model,
           "test_scalp_pass_rate": res["test_scalp_pass_rate"],
           "test_not_scalp_reject_rate": res["test_not_scalp_reject_rate"]}
    json.dump(cfg, open(A / "gate_config.json", "w"), indent=2)
    json.dump(res, open(R / "gate_results.json", "w"), indent=2)
    print("saved artifacts/gate.onnx, artifacts/gate_config.json, reports/gate_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/split_final.csv")
    ap.add_argument("--model", default="mobilenetv3_small_100")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-false-reject", type=float, default=0.01)
    ap.add_argument("--neg-dir", default="data/not_scalp")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-caltech", action="store_true", help="use only data/not_scalp/")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 64 images per split")
    main(ap.parse_args())
