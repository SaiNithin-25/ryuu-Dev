@echo off
echo ================================================
echo     Starting RyuuGPT v3 Training
echo ================================================
echo.

set MODE=%1
if "%MODE%"=="" set MODE=full
echo Mode: %MODE%
echo.

REM -------------------------------
REM Performance knobs (edit as needed)
REM -------------------------------
set NUM_WORKERS=4
set PREFETCH=4
set MATMUL_PREC=high
REM Uncomment to try torch.compile (can be slower to start)
REM set USE_COMPILE=--compile --compile_mode max-autotune

REM -------------------------------
REM Activate environment
REM -------------------------------
call Dev\Scripts\activate

REM -------------------------------
REM CUDA safety for long runs
REM -------------------------------
set PYTORCH_ALLOC_CONF=expandable_segments:True

if /I "%MODE%"=="small" (
  python training/train_v3.1.py ^
    --data_dir data/open/Ryuu_Developer_v1/tokenized ^
    --save_dir checkpoints/v3_small ^
    --log_dir runs/v3_small ^
    --batch_size 8 ^
    --grad_accum 2 ^
    --context_size 512 ^
    --n_layer 8 ^
    --n_head 8 ^
    --n_embd 512 ^
    --dropout 0.1 ^
    --lr 4e-4 ^
    --warmup_steps 1000 ^
    --max_steps 20000 ^
    --eval_interval 1000 ^
    --use_bf16 ^
    --enable_checkpointing ^
    --num_workers %NUM_WORKERS% ^
    --pin_memory ^
    --persistent_workers ^
    --prefetch_factor %PREFETCH% ^
    --cudnn_benchmark ^
    --matmul_precision %MATMUL_PREC% ^
    --tokenizer_path tokenizer/bpe_tokenizer_postproc.json ^
    --vocab_size 30000 ^
    --seed 42 ^
    --compile_mode default ^
    --shuffle_mode shard ^
    --log_gpu ^
    %USE_COMPILE%
) else if /I "%MODE%"=="medium" (
  python training/train_v3.1.py ^
    --data_dir data/open/Ryuu_Developer_v1/tokenized ^
    --save_dir checkpoints/v3_medium ^
    --log_dir runs/v3_medium ^
    --batch_size 12 ^
    --grad_accum 18 ^
    --context_size 768 ^
    --n_layer 12 ^
    --n_head 8 ^
    --n_embd 640 ^
    --dropout 0.1 ^
    --lr 3e-4 ^
    --warmup_steps 2000 ^
    --max_steps 40000 ^
    --eval_interval 1000 ^
    --use_bf16 ^
    --enable_checkpointing ^
    --num_workers %NUM_WORKERS% ^
    --pin_memory ^
    --persistent_workers ^
    --prefetch_factor %PREFETCH% ^
    --cudnn_benchmark ^
    --matmul_precision %MATMUL_PREC% ^
    --tokenizer_path tokenizer/bpe_tokenizer_postproc.json ^
    --vocab_size 30000 ^
    --seed 42 ^
    --compile_mode default ^
    --shuffle_mode shard ^
    --log_gpu ^
    %USE_COMPILE%
) else (
  python training/train_v3.1.py ^
    --data_dir data/open/Ryuu_Developer_v1/tokenized ^
    --save_dir checkpoints/v3_reasoning ^
    --log_dir runs/v3_reasoning ^
    --batch_size 4 ^
    --grad_accum 16 ^
    --context_size 768 ^
    --n_layer 12 ^
    --n_head 12 ^
    --n_embd 768 ^
    --dropout 0.1 ^
    --lr 2e-4 ^
    --warmup_steps 1500 ^
    --max_steps 52240 ^
    --eval_interval 1000 ^
    --use_bf16 ^
    --enable_checkpointing ^
    --enable_reasoning_head ^
    --reasoning_loss_weight 0.1 ^
    --reasoning_warmup_steps 3000 ^
    --reasoning_full_steps 12000 ^
    --reasoning_num_layers 4 ^
    --reasoning_dim 16 ^
    --reasoning_dropout 0.1 ^
    --num_workers %NUM_WORKERS% ^
    --pin_memory ^
    --persistent_workers ^
    --prefetch_factor %PREFETCH% ^
    --matmul_precision %MATMUL_PREC% ^
    --tokenizer_path tokenizer/bpe_tokenizer_postproc.json ^
    --vocab_size 30000 ^
    --seed 42 ^
    --compile_mode default ^
    --shuffle_mode shard ^
    --log_gpu ^
    %USE_COMPILE%

)

echo.
echo ================================================
echo           Training Finished
echo ================================================
pause