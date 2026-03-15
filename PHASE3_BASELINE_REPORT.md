# PHASE 3 BASELINE REPORT

Date: 2026-02-22
Environment: `cuda\Scripts\python.exe` (Windows)

## Scope
- Added end-to-end inference smoke test using latest available checkpoint.
- Added tokenizer consistency checks (special tokens, roundtrip, EOS behavior).
- Ran mini-train + eval baseline using `training/train_v3.1.py` via smoke harness.

## New Tests Added
- `testing/test_inference_smoke.py`
- `testing/test_tokenizer_consistency.py`

## Commands Run
1. `cuda\Scripts\python.exe -m testing.test_tokenizer_consistency`
2. `cuda\Scripts\python.exe -m testing.test_inference_smoke`
3. `cuda\Scripts\python.exe -m testing.test_trainer_smoke --data_dir data/phase3_train_data --ckpt_dir checkpoints/phase3_train_smoke --logs_dir runs/phase3_train_smoke`

## Baseline Training Metrics
- Checkpoint dir: `checkpoints/phase3_train_smoke`
- Best checkpoint: `checkpoints/phase3_train_smoke/ckpt_best.pt`
- Best step: `20`
- Best validation loss: `10.072714646657309`

## Smoke Validation Results
- Tokenizer consistency: PASS
- Inference smoke: PASS
- Trainer smoke (`train_v3.1.py`): PASS
- Existing checks/tests rerun in this phase: PASS (except expected non-deterministic text quality in tiny-model generation output)

## Notes
- Inference smoke reserves prompt headroom for small-context checkpoints to avoid generation truncation false negatives.
- Smoke shard creation now handles Windows file locks by falling back to a temporary directory and copy-based test shard preparation.
