import json
from pathlib import Path

from src.dataset import get_loaders


def save_classes(classes, path="artifacts/labels.json"):
    """Persist ImageFolder's class-index order for inference and export."""
    classes = list(classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes must be a non-empty sequence of unique names")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(classes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_training_data(**loader_kwargs):
    """Build loaders and save the exact ImageFolder class order used for training."""
    sets, loaders = get_loaders(**loader_kwargs)
    save_classes(sets["train"].classes)
    return sets, loaders
