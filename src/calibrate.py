import argparse, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

R = Path("reports")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def nll(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def fit_temperature(logits, y):
    z, t = torch.tensor(logits), torch.tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def pick_threshold(conf, correct, target):
    order = np.argsort(-conf)
    cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(cum >= target)[0]
    return None if len(ok) == 0 else float(conf[order][ok.max()])


def reliability(ax, probs, y, title, bins=10):
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(xs, ys, "o-")
    ax.set_title(title); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")


def main(a):
    lv, lt = np.load(R / f"logits_val_{a.name}.npy"), np.load(R / f"logits_test_{a.name}.npy")
    pv, pt = pd.read_csv(R / f"preds_val_{a.name}.csv"), pd.read_csv(R / f"preds_test_{a.name}.csv")
    yv, yt = pv.label.to_numpy(), pt.label.to_numpy()

    T = fit_temperature(lv, yv)
    print(f"temperature T = {T:.3f}  (below 1 means the model was under-confident)")
    rows = []
    for split, lg, y in [("val", lv, yv), ("test", lt, yt)]:
        b, c = softmax(lg), softmax(lg / T)
        rows.append({"split": split, "ECE before": ece(b, y), "ECE after": ece(c, y),
                     "NLL before": nll(b, y), "NLL after": nll(c, y)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # reject option: threshold chosen on validation only
    cv, ct = softmax(lv / T).max(1), softmax(lt / T).max(1)
    corv, cort = lv.argmax(1) == yv, lt.argmax(1) == yt
    thr = pick_threshold(cv, corv, a.target)
    if thr is None:
        print(f"validation accuracy never reaches {a.target}; lower --target")
        return
    low = (pd.qcut(pt.nn_sim, 3, labels=False) == 0).to_numpy()
    acc_t = cort[ct >= thr].mean()
    print(f"\nreject threshold {thr:.3f} (chosen on val for accuracy >= {a.target})")
    print(f"  val : coverage {(cv >= thr).mean():.3f}  accuracy on accepted {corv[cv >= thr].mean():.4f}")
    print(f"  test: coverage {(ct >= thr).mean():.3f}  accuracy on accepted {acc_t:.4f}  "
          f"(overall {cort.mean():.4f})")
    print(f"  test, least-similar third: coverage {(ct[low] >= thr).mean():.3f}  "
          f"accuracy on accepted {cort[low][ct[low] >= thr].mean():.4f}  (overall {cort[low].mean():.4f})")

    # figures
    Path("artifacts").mkdir(exist_ok=True); (R / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    reliability(ax[0], softmax(lt), yt, "test, before"); reliability(ax[1], softmax(lt / T), yt, "test, after")
    plt.tight_layout(); plt.savefig(R / "figures" / f"reliability_{a.name}.png", dpi=200); plt.close()
    plt.figure(figsize=(5, 4))
    for label, m in [("all test", np.ones(len(yt), bool)), ("least-similar third", low)]:
        order = np.argsort(-ct[m])
        plt.plot(np.arange(1, m.sum() + 1) / m.sum(), np.cumsum(cort[m][order]) / np.arange(1, m.sum() + 1), label=label)
    plt.xlabel("coverage"); plt.ylabel("accuracy on accepted"); plt.legend(); plt.tight_layout()
    plt.savefig(R / "figures" / f"coverage_{a.name}.png", dpi=200); plt.close()

    json.dump({"model": a.name, "temperature": T, "reject_threshold": thr, "target": a.target},
              open("artifacts/calibration.json", "w"), indent=2)
    print("saved artifacts/calibration.json and two figures in reports/figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", type=float, default=0.98)
    main(ap.parse_args())