import argparse, json, random, time
from pathlib import Path

import mlflow
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from src.dataset import get_loaders


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, device, max_batches=None):
    model.eval()
    crit = nn.CrossEntropyLoss()
    ys, ps, loss_sum = [], [], 0.0
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * len(y)
        ys += y.tolist()
        ps += out.argmax(1).tolist()
    ys, ps = np.array(ys), np.array(ps)
    macro_f1 = f1_score(ys, ps, average="macro", zero_division=0)
    return loss_sum / len(ys), macro_f1, float((ys == ps).mean())


def make_scaler(use_amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):            # older PyTorch versions
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def main(cfg_path, seed):
    cfg = yaml.safe_load(open(cfg_path))
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    max_b = cfg.get("max_batches")                  # only set in the smoke-test config
    print(f"device: {device}")

    # data
    sets, loaders = get_loaders(img_size=cfg["img_size"], batch_size=cfg["batch_size"],
                                num_workers=cfg.get("num_workers", 0))
    classes = sets["train"].classes
    print("images:", {k: len(v) for k, v in sets.items()})
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
    json.dump(classes, open("artifacts/labels.json", "w"))

    # model: pretrained backbone with a new head for our classes
    model = timm.create_model(cfg["model"], pretrained=True, num_classes=len(classes)).to(device)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    # class-weighted loss, computed from the (uneven) training counts
    counts = np.bincount(sets["train"].targets, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts),
                           dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])

    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": cfg["lr_head"]},
         {"params": backbone_params, "lr": cfg["lr_backbone"]}],
        weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = make_scaler(use_amp)

    ckpt = f"artifacts/checkpoints/{cfg['run_name']}_seed{seed}.pt"
    best_f1, bad = -1.0, 0

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=f"{cfg['run_name']}_seed{seed}"):
        mlflow.log_params({**cfg, "seed": seed, "device": device,
                           "train_images": len(sets["train"])})
        mlflow.log_dict({c: int(n) for c, n in zip(classes, counts)}, "train_class_counts.json")
        mlflow.log_artifact(cfg_path)

        for epoch in range(cfg["epochs"]):
            t0 = time.time()
            for p in backbone_params:               # two-stage fine-tuning: head first
                p.requires_grad = epoch >= cfg["frozen_epochs"]

            model.train()
            run_loss, n_seen = 0.0, 0
            for bi, (x, y) in enumerate(loaders["train"]):
                if max_b and bi >= max_b:
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=device, enabled=use_amp):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item() * len(y)
                n_seen += len(y)
            sched.step()

            val_loss, val_f1, val_acc = run_eval(model, loaders["val"], device, max_b)
            mlflow.log_metrics({"train_loss": run_loss / n_seen, "val_loss": val_loss,
                                "val_macro_f1": val_f1, "val_acc": val_acc}, step=epoch)
            print(f"epoch {epoch}: train_loss={run_loss / n_seen:.4f} "
                  f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)")

            if val_f1 > best_f1:                    # keep the best checkpoint on validation
                best_f1, bad = val_f1, 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    print("early stopping")
                    break

        mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.log_artifact(ckpt)
        print(f"best val macro-F1: {best_f1:.4f}  ->  {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.config, a.seed)