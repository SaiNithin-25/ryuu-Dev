# RELEASE CHECKLIST

Date baseline: 2026-02-22
Project: RyuuAI

## Release Gates (must all pass)
- [x] `run_all_checks.bat` exits with code 0 on local CUDA environment.
- [x] `testing.test_checkpoint_matrix` passes on all required checkpoint formats.
- [x] `testing.test_inference_deterministic` passes (repeatable greedy generation).
- [x] `testing.test_inference_smoke` passes (basic generation path).
- [x] `testing.test_trainer_smoke` passes using `training/train_v3.1.py`.

## Artifact Validation
- [x] Required checkpoint exists for release target (for example `checkpoints/v3_reasoning/ckpt_step19000.pt` or selected final).
- [x] Tokenizer exists and matches model vocab assumptions: `tokenizer/bpe_tokenizer_postproc.json`.
- [x] Prompt protocol exists: `prompts/ryuu_dev_protocol.txt`.

## Runtime Validation
- [x] Service import test passes: `python -c "import experts.ryuu_dev_service; print('ok')"`.
- [x] End-to-end inference responds with non-empty output.
- [x] No blocking errors in logs for checkpoint load, tokenizer load, or generation.

## CI Validation
- [ ] GitHub Actions workflow `Key Checks (CPU)` passes on latest commit.
- [x] CPU skip list is current with any newly-added GPU/checkpoint-heavy tests.

## Final Release Steps
- [x] Update release notes with checkpoint/tag name.
- [x] Record exact command outputs in release notes or issue.
- [ ] Create git tag for release (example: `vX.Y.Z`). (Skipped by request)
- [x] Freeze this checklist and `KNOWN_GOOD_CONFIG.md` for traceability.
