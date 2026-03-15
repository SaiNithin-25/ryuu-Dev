@echo off
setlocal enabledelayedexpansion

echo ================================================
echo       RyuuGPT v3  DPO TRAINING (PHASE 3)
echo ================================================
echo.

REM -------------------------------
REM Activate environment
REM -------------------------------
call Dev\Scripts\activate

REM -------- CONFIGURE PATHS HERE ------------------

REM Path to your generated DPO dataset
set DPO_DATA=data\dpo\ryuu_dpo.jsonl

REM Path to your SFT Reasoning Phase-2 best checkpoint
set CKPT=checkpoints\v3_reasoning\ckpt_best.pt

REM Output directory for DPO finetuned model
set SAVE_DIR=checkpoints\v3_dpo

REM Batch size and steps
set BS=1
set STEPS=3000
set LR=1e-5
set BETA=0.1

REM ------------------------------------------------

echo Using dataset: %DPO_DATA%
echo Using checkpoint: %CKPT%
echo Saving to: %SAVE_DIR%
echo Batch size: %BS%
echo Steps: %STEPS%
echo Learning rate: %LR%
echo Beta: %BETA%
echo.

REM Ensure directories exist
if not exist %SAVE_DIR% (
    mkdir %SAVE_DIR%
)

REM -------- RUN DPO TRAINING ----------------------

python training\dpo\dpo_trainer.py ^
  --data %DPO_DATA% ^
  --ckpt %CKPT% ^
  --save_dir %SAVE_DIR% ^
  --batch_size %BS% ^
  --max_steps %STEPS% ^
  --lr %LR% ^
  --beta %BETA%

echo.
echo ================================================
echo       DPO TRAINING COMPLETED SUCCESSFULLY 
echo ================================================
echo.

pause
