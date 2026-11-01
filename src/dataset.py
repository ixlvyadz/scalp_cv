from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def get_transforms(img_size=224):
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),   # keep mild: color is diagnostic
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

def _filter_imagefolder(image_set, list_path, root):
    """Keep only paths listed in a newline-delimited file, relative to root."""
    list_path = Path(list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Clean test list not found: {list_path}")

    root = Path(root).resolve()
    allowed = {
        Path(line.strip().replace("\\", "/")).as_posix()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    filtered = []
    found = set()
    for image_path, target in image_set.samples:
        relative = Path(image_path).resolve().relative_to(root).as_posix()
        if relative in allowed:
            filtered.append((image_path, target))
            found.add(relative)

    missing = allowed - found
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise ValueError(f"{len(missing)} listed clean test images were not found under {root}: {preview}")
    if not filtered:
        raise ValueError(f"Clean test list {list_path} matched no images under {root}")

    image_set.samples = filtered
    image_set.imgs = filtered
    image_set.targets = [target for _, target in filtered]
    return image_set


def get_loaders(root="data", img_size=224, batch_size=32, num_workers=2,
                clean_test_list="reports/clean_test_list.txt"):
    train_tf, eval_tf = get_transforms(img_size)
    sets = {s: datasets.ImageFolder(f"{root}/{s}", transform=(train_tf if s == "train" else eval_tf))
            for s in ["train", "val", "test"]}
    assert sets["train"].classes == sets["val"].classes == sets["test"].classes, "class folders differ!"
    sets["test_clean"] = _filter_imagefolder(
        datasets.ImageFolder(f"{root}/test", transform=eval_tf), clean_test_list, f"{root}/test"
    )
    assert sets["test_clean"].classes == sets["test"].classes, "class folders differ!"
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"),
                             num_workers=num_workers, pin_memory=True)
               for s, d in sets.items()}
    return sets, loaders
