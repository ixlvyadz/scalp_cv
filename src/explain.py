import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from src.dataset import get_loaders

R = Path("reports")
FR = np.linspace(0, 1, 21)                       # fraction of pixels removed or revealed
LAYERS = {"mobilenetv3_large_100": lambda m: m.blocks[-1],
          "resnet50": lambda m: m.layer4[-1],
          "convnext_tiny": lambda m: m.stages[-1]}
SHOW = T.Compose([T.Resize(int(224 * 1.14)), T.CenterCrop(224)])


def auc(c):
    return float(np.sum((c[1:] + c[:-1]) / 2 * np.diff(FR)))


def curves(model, x, saliency, cls, device):
    """Probability of class `cls` while deleting / inserting pixels in order of saliency."""
    H, W = saliency.shape
    noisy = saliency + 1e-6 * np.random.rand(H, W)               # break ties
    rank = np.empty(H * W, dtype=np.int64)
    rank[np.argsort(-noisy.ravel())] = np.arange(H * W)          # 0 = most important pixel
    rank = torch.tensor(rank.reshape(H, W))
    blur = gaussian_blur(x, kernel_size=51, sigma=20.0)
    dele, ins = [], []
    for f in FR:
        top = rank < int(round(f * H * W))
        dele.append(torch.where(top, torch.zeros_like(x), x))    # deletion: remove top pixels
        ins.append(torch.where(top, x, blur))                    # insertion: reveal top pixels
    with torch.no_grad():
        p = torch.softmax(model(torch.stack(dele + ins).to(device)), 1)[:, cls].cpu().numpy()
    return p[:len(FR)], p[len(FR):]


def main(a):
    global device
    device = a.device
    np.random.seed(0); torch.manual_seed(0)
    rng = np.random.default_rng(0)
    sets, _ = get_loaders(num_workers=0)
    ds = sets["test"]
    classes = ds.classes
    pdf = pd.read_csv(R / f"preds_test_{a.name}.csv")
    assert (pdf.path.to_numpy() == np.array([s[0] for s in ds.samples])).all()

    model = timm.create_model(a.model, pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.to(device).eval()
    cam = GradCAMPlusPlus(model=model, target_layers=[LAYERS[a.model](model)])

    def explain(i):
        x, _ = ds[i]
        with torch.no_grad():
            out = torch.softmax(model(x[None].to(device)), 1)[0]
        cls = int(out.argmax())
        g = cam(input_tensor=x[None].to(device), targets=[ClassifierOutputTarget(cls)])[0]
        return x, cls, float(out[cls]), g

    def overlay(i, g):
        rgb = np.asarray(SHOW(Image.open(pdf.path[i]).convert("RGB")), dtype=np.float32) / 255
        return show_cam_on_image(rgb, g, use_rgb=True)

    one = pdf.groupby("group").sample(1, random_state=0)         # one image per group
    (R / "figures").mkdir(exist_ok=True)

    # 1. gallery: top two rows correct, bottom two rows wrong
    right = one[one.label == one.pred].sample(frac=1, random_state=0).head(8).index
    wrong = one[one.label != one.pred].sample(frac=1, random_state=0).head(8).index
    fig, axes = plt.subplots(4, 4, figsize=(12, 13))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), list(right) + list(wrong)):
        _, cls, conf, g = explain(i)
        ax.imshow(overlay(i, g))
        ax.set_title(f"true: {classes[pdf.label[i]]}\npred: {classes[cls]} ({conf:.2f})", fontsize=8)
    plt.tight_layout(); plt.savefig(R / "figures" / f"gradcam_{a.name}.png", dpi=150); plt.close()

    # 2. faithfulness on about per_class images per class, plus 30 overlays for manual review
    pick = one.sample(frac=1, random_state=1).groupby("label").head(a.per_class)
    review = set(pick.groupby("label").head(3).index)
    rdir = R / "figures" / f"gradcam_review_{a.name}"
    rdir.mkdir(exist_ok=True)
    rows, cc = [], {"del": [], "ins": [], "del_r": [], "ins_r": []}
    for i in tqdm(pick.index, desc="faithfulness"):
        x, cls, conf, g = explain(i)
        d, n = curves(model, x, g, cls, device)
        dr, nr = curves(model, x, rng.random(g.shape), cls, device)
        cc["del"].append(d); cc["ins"].append(n); cc["del_r"].append(dr); cc["ins_r"].append(nr)
        rows.append({"path": pdf.path[i], "true": classes[pdf.label[i]], "pred": classes[cls],
                     "deletion_cam": auc(d), "insertion_cam": auc(n),
                     "deletion_random": auc(dr), "insertion_random": auc(nr)})
        if i in review:
            Image.fromarray(overlay(i, g)).save(rdir / f"{len(list(rdir.glob('*.png'))):02d}_{classes[pdf.label[i]]}_as_{classes[cls]}.png")
    df = pd.DataFrame(rows)
    df.to_csv(R / f"faithfulness_{a.name}.csv", index=False)

    print(f"\n{len(df)} test images, one per group")
    print(f"deletion AUC  (lower is better): Grad-CAM++ {df.deletion_cam.mean():.3f}   random {df.deletion_random.mean():.3f}")
    print(f"insertion AUC (higher is better): Grad-CAM++ {df.insertion_cam.mean():.3f}   random {df.insertion_random.mean():.3f}")
    for name, diff in [("deletion advantage (random - CAM)", df.deletion_random - df.deletion_cam),
                       ("insertion advantage (CAM - random)", df.insertion_cam - df.insertion_random)]:
        b = rng.choice(diff.to_numpy(), (1000, len(diff))).mean(1)
        print(f"{name}: {diff.mean():+.3f}  95% interval [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for k, lab, col in [("del", "Grad-CAM++", "C0"), ("del_r", "random", "gray")]:
        ax[0].plot(FR, np.mean(cc[k], 0), label=lab, color=col)
    for k, lab, col in [("ins", "Grad-CAM++", "C0"), ("ins_r", "random", "gray")]:
        ax[1].plot(FR, np.mean(cc[k], 0), label=lab, color=col)
    ax[0].set_title("deletion (lower is better)"); ax[1].set_title("insertion (higher is better)")
    for x_ in ax:
        x_.set_xlabel("fraction of pixels"); x_.set_ylabel("probability of predicted class"); x_.legend()
    plt.tight_layout(); plt.savefig(R / "figures" / f"faithfulness_{a.name}.png", dpi=200); plt.close()
    print("saved gallery, curves and the review overlays in reports/figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-class", type=int, default=12, dest="per_class")
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args())