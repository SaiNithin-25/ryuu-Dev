# PHASE 5 LONG BASELINE REPORT

Date: 2026-02-22

## Run Command
`cuda\\Scripts\\python.exe training/train_v3.1.py --data_dir data/phase3_train_data --save_dir checkpoints/phase5_long_baseline --log_dir runs/phase5_long_baseline --batch_size 2 --grad_accum 1 --max_steps 500 --eval_interval 50 --warmup_steps 20 --context_size 32 --n_layer 2 --n_head 2 --n_embd 128 --dropout 0.0`

## Artifacts
- Save dir: `checkpoints/phase5_long_baseline`
- Log dir: `runs/phase5_long_baseline`
- Best checkpoint: `checkpoints/phase5_long_baseline/ckpt_best.pt`
- Step checkpoints written: 10 (`ckpt_step50.pt` through `ckpt_step500.pt`)

## Metrics
- Best step: `500`
- Best validation loss: `6.5971198081970215`

## Comparison vs Previous Smoke Baseline
Previous baseline (Phase 3 smoke):
- Best val loss: `10.072714646657309`
- Best step: `20`

Current long baseline:
- Best val loss: `6.5971198081970215`
- Best step: `500`

Absolute improvement (lower is better):
- `10.072714646657309 - 6.5971198081970215 = 3.475594838460287`

Relative improvement:
- `(3.475594838460287 / 10.072714646657309) * 100 = 34.50%`

## Conclusion
The longer baseline run improved validation loss substantially versus the smoke baseline (about 34.50% relative improvement), and training remained stable through step 500.
