import numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

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

class SplitDataset(Dataset):
    def __init__(self, df, class_to_idx, transform):
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.samples = [(p, class_to_idx[l]) for p, l in zip(df.path, df.label)]
        self.targets = [y for _, y in self.samples]
        self.groups = df.group.to_numpy()                    # for group-aware confidence intervals
        self.nn_sim = df.nn_train_sim.to_numpy()             # closeness to the nearest train image (NaN for train)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y

def get_loaders(csv="reports/split_final.csv", img_size=224, batch_size=32, num_workers=2):
    df = pd.read_csv(csv)
    classes = sorted(df.label.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    train_tf, eval_tf = get_transforms(img_size)
    sets = {s: SplitDataset(df[df.final == s].reset_index(drop=True), c2i,
                            train_tf if s == "train" else eval_tf)
            for s in ["train", "val", "test"]}
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"),
                             num_workers=num_workers, pin_memory=True)
               for s, d in sets.items()}
    return sets, loaders