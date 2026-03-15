# KNOWN GOOD CONFIG

Date validated: 2026-02-22

## Environment
- OS: Windows 10
- Python: `cuda\\Scripts\\python.exe`
- Device: CUDA (`NVIDIA GeForce RTX 4050 Laptop GPU`)

## Key Commands (validated)
- Full local check suite:
  - `run_all_checks.bat`
- Direct runner:
  - `cuda\\Scripts\\python.exe testing\\run_key_checks.py --python cuda\\Scripts\\python.exe`

## Validated Test Set
- checks.check_power
- checks.check_shards
- checks.check_ckpt
- checks.simple_check
- checks.test_config
- testing.test_tok
- testing.test_tokenizer_consistency
- testing.test_protocol_inference
- testing.test2
- testing.sanity_check_ryuugpt_v4
- testing.test_checkpoint_matrix
- testing.test_inference_smoke
- testing.test_inference_deterministic
- testing.reasoning_test
- testing.test_trainer_smoke

## Baseline Training Smoke Result
- Script: `training/train_v3.1.py` (invoked via `testing/test_trainer_smoke.py`)
- Best validation loss observed in Phase 3 baseline run:
  - `10.072714646657309` at step `20`
- Baseline checkpoint:
  - `checkpoints/phase3_train_smoke/ckpt_best.pt`

## Checkpoint Compatibility (validated)
- `checkpoints/test_smoke/ckpt_best.pt`
- `checkpoints/phase3_train_smoke/ckpt_best.pt`
- `checkpoints/v3_reasoning/ckpt_step19000.pt`

## Notes
- Inference quality in smoke tests is not a quality benchmark; smoke tests validate runtime stability.
- Windows file locks can occur for synthetic shard cleanup; smoke test includes fallback temp directory handling.
