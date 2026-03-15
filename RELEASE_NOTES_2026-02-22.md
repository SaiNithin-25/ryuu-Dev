# RELEASE NOTES - 2026-02-22

Project: RyuuAI
Release type: Stabilization and hardening

## Summary
This release finalizes training/inference runtime stability, checkpoint compatibility, and test automation for local and CI workflows.

## Major Improvements
- Unified checkpoint compatibility handling across training/inference utilities.
  - Supports `model`, `model_state`, `state_dict`, and raw state dict formats.
- Fixed generation runtime bug in `model/Ryuu_gpt.py` (`Tensor * dict` reasoning steering issue).
- Added robust inference and tokenizer smoke tests.
- Added deterministic generation test for repeatable greedy decoding.
- Added checkpoint matrix test to validate loading of multiple checkpoint formats/targets.
- Added one-command test runner and batch entry point.

## New/Updated Validation Tools
- One-command local suite: `run_all_checks.bat`
- Python runner: `testing/run_key_checks.py`
- New tests:
  - `testing/test_inference_smoke.py`
  - `testing/test_inference_deterministic.py`
  - `testing/test_checkpoint_matrix.py`
  - `testing/test_tokenizer_consistency.py`

## Final Gate Result
- Command: `run_all_checks.bat`
- Result: `15/15` passed
- Exit code: `0`

## Checkpoints Validated in Matrix
- `checkpoints/test_smoke/ckpt_best.pt`
- `checkpoints/phase3_train_smoke/ckpt_best.pt`
- `checkpoints/v3_reasoning/ckpt_step19000.pt`

## Baseline Training Progress
Long baseline run (`500` steps) using `training/train_v3.1.py`:
- Best validation loss: `6.5971198081970215`
- Previous smoke baseline: `10.072714646657309`
- Relative improvement: `34.50%`
- Detailed report: `PHASE5_LONG_BASELINE_REPORT.md`

## CI
Added workflow: `.github/workflows/checks.yml`
- Name: `Key Checks (CPU)`
- Runs CPU-safe subset of key checks/tests on push and pull request.

## Notes
- Git tagging intentionally skipped for this release cycle by request.
