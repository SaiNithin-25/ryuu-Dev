@echo off
REM ============================================
REM RyuuGPT DPO Dataset Generator 
REM ============================================

REM Activate project environment
call Dev\Scripts\activate

REM Sanity check
python -c "import torch; print('Torch OK:', torch.__version__)"

set CKPT=checkpoints\v3_reasoning\ckpt_best.pt
set TOKENIZER=tokenizer\bpe_tokenizer_postproc.json
set OUT=data\dpo\ryuu_dpo.jsonl
set NUM_SAMPLES=5000
set BATCH_SIZE=6
set MAX_TOKENS=128

REM Using FAST batch version (3-5x faster)
python training\dpo\dpo_dataset_fast.py ^
  --ckpt %CKPT% ^
  --tokenizer %TOKENIZER% ^
  --out %OUT% ^
  --num_samples %NUM_SAMPLES% ^
  --batch_size %BATCH_SIZE% ^
  --max_new_tokens %MAX_TOKENS% ^
  --temperature 0.8 ^
  --top_k 50

pause
