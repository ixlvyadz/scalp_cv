# "Not a scalp photo" gate

**Problem.** The classifier always picks one of its 10 conditions, because it has no
"none of these" option. A photo of a laptop came back as *Head Lice*. Temperature scaling
and the reject threshold don't fix this: the model can be confidently wrong on photos that
look nothing like the training data (a random-noise image scored 81% Folliculitis).

**Fix.** A small second model (MobileNetV3-Small, about 6 MB) answers one question first:
*is this a photo of the scalp or hair?* Only photos that pass go to the classifier.
The classifier itself and its reported results are unchanged.

## Train it (about 10-20 minutes on a GPU)

```
python -m src.train_gate --smoke     # quick check that everything runs
python -m src.train_gate             # real run
```

- **Scalp photos:** the same train/val/test split as the classifier (`reports/split_final.csv`).
- **Not-scalp photos:** Caltech-101 everyday objects (downloaded once into `data/`, face classes
  excluded), plus anything in `data/not_scalp/`. **Add 100-300 of your own phone photos** there
  (laptops, desks, screens, rooms, hands, arms, pets, food, walls): those are the photos real
  users will actually try.
- **Threshold:** chosen on validation so at most 1% of real scalp photos are rejected
  (`--max-false-reject 0.01`).

It writes `artifacts/gate.onnx`, `artifacts/gate_config.json` and `reports/gate_results.json`
(test pass/reject rates, and which scalp test photos it wrongly rejects, by class).

## What uses it

- `src/predict.py` now also returns `is_scalp` and `scalp_score` (via `src/gate.py`).
  If `artifacts/gate.onnx` is missing, `is_scalp` is always `True`, so nothing else changes.
- `app/ui.py` (Streamlit) stops with "This does not look like a photo of the scalp or hair".
- The Android app (`scalp_check_mobile_app/`) copies the gate automatically and shows
  "This doesn't look like a scalp photo".

## Check before you rely on it

1. `reports/gate_results.json`: `test_scalp_pass_rate` should be about 0.99 and
   `test_not_scalp_reject_rate` as high as possible.
2. Look at `test_scalp_rejected_files`: if one class loses many photos, add more varied
   negatives or lower `--max-false-reject`.
3. Try 20+ everyday objects with your phone in the app.
