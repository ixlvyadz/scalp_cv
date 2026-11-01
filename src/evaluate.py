import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.inference_mode()
def evaluate_test_sets(model, loaders, device):
    """Evaluate the same model on the official and cleaned test loaders.

    ``loaders`` is the mapping returned by :func:`src.dataset.get_loaders`.
    Returns accuracy, macro F1, and sample count for each test set.
    """
    results = {}
    was_training = model.training
    model.eval()
    try:
        for split in ("test", "test_clean"):
            if split not in loaders:
                raise KeyError(f"Missing {split!r} loader; use get_loaders()")
            y_true, y_pred = [], []
            for images, labels in loaders[split]:
                logits = model(images.to(device))
                y_true.extend(labels.tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            results[split] = {
                "n": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
    finally:
        model.train(was_training)
    return results
