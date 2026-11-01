import numpy as np, pandas as pd, torch, timm, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

DATA, OUT = Path("data"), Path("reports")
EXT = {".jpg", ".jpeg", ".png"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "resnet50" if dev == "cuda" else "resnet18"
print("device:", dev, "| model:", MODEL)

rows = [(s, p.parent.name, p.as_posix()) for s in ["train", "val", "test"]
        for p in sorted((DATA / s).rglob("*")) if p.suffix.lower() in EXT]
df = pd.DataFrame(rows, columns=["split", "label", "path"])

cache = OUT / f"emb_{MODEL}.npz"
if cache.exists():
    z = np.load(cache); E, Em = z["E"], z["Em"]
else:
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    E, Em = [], []
    with torch.no_grad():
        for a in tqdm(range(0, len(df), 64), desc="embedding"):
            ims = [Image.open(p).convert("RGB") for p in df.path.iloc[a:a + 64]]
            x = torch.stack([tf(im) for im in ims]).to(dev)
            xm = torch.stack([tf(ImageOps.mirror(im)) for im in ims]).to(dev)
            for out, feats in ((E, model(x)), (Em, model(xm))):
                out.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype(np.float16))
    E, Em = np.concatenate(E), np.concatenate(Em)
    np.savez(cache, E=E, Em=Em)

# nearest train image for every val/test image (mirror-aware)
tr = np.where(df.split == "train")[0]
Et = E[tr].astype(np.float32).T
res = []
for i in tqdm(np.where(df.split != "train")[0], desc="nearest train image"):
    s1, s2 = E[i].astype(np.float32) @ Et, Em[i].astype(np.float32) @ Et
    j1, j2 = s1.argmax(), s2.argmax()
    res.append((i, tr[j1], float(s1[j1]), "original") if s1[j1] >= s2[j2]
               else (i, tr[j2], float(s2[j2]), "mirror"))
P = pd.DataFrame(res, columns=["i", "j", "sim", "via"])
P["split"] = df.split.values[P.i]
P.to_csv(OUT / "emb_nearest_train.csv", index=False)

print("\nnearest-train similarity, by split:")
print(pd.crosstab(pd.cut(P.sim, [0, .6, .65, .7, .75, .8, .85, .9, .95, 1.01]), P.split))

sheetdir = OUT / "figures" / "leak_emb"
sheetdir.mkdir(parents=True, exist_ok=True)
for lo, hi in [(.95, 1.01), (.90, .95), (.85, .90), (.80, .85), (.75, .80)]:
    sub = P[(P.sim >= lo) & (P.sim < hi)]
    if sub.empty:
        continue
    rows_ = list(sub.sample(min(16, len(sub)), random_state=0).itertuples())
    for k in range(0, len(rows_), 8):
        chunk = rows_[k:k + 8]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 3 * len(chunk)), squeeze=False)
        for r, (a, b) in zip(chunk, axes):
            a.imshow(Image.open(df.path[r.i])); a.set_title(f"{r.split}, sim={r.sim:.3f}", fontsize=7)
            b.imshow(Image.open(df.path[r.j])); b.set_title(f"nearest train ({r.via})", fontsize=7)
            a.axis("off"); b.axis("off")
        plt.tight_layout()
        plt.savefig(sheetdir / f"sim_{lo:.2f}_{k // 8}.png", dpi=150)
        plt.close()
print("sheets in", sheetdir)