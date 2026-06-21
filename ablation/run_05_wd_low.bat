@echo off
cd /d "%~dp0\.."
set KMP_DUPLICATE_LIB_OK=TRUE
set USE_LIBUV=0
call scripts\load_env.cmd
if "%HF_TOKEN%"=="" (
  echo ERROR: HF_TOKEN not set. Create a .env file or set the environment variable.
  echo See docs\ENV.md for instructions.
  exit /b 1
)
REM Lower weight-decay ablation: weight_decay 0.4
if not exist fineweb_data\fineweb_train.pt (
  echo Preparing data in fineweb_data...
  python prepare_data.py --train_tokens 10000000 --val_tokens 1000000 --local_dir fineweb_data --skip-verify
  if errorlevel 1 (
    echo prepare_data failed. Exiting.
    exit /b 1
  )
)
if not exist fineweb_data\fineweb_train.pt (
  echo ERROR: fineweb_data\fineweb_train.pt still missing after prepare_data.py
  exit /b 1
)
echo.
echo Running low weight-decay experiment...
echo.
python tiny/train.py ^
  --run-name wd_low_10M_16K ^
  --wandb_entity i-learn ^
  --no_torch_compile ^
  --n_layer 4 --n_embd 512 --n_head 8 ^
  --device-batch-size 2 --num-epochs 16 ^
  --no-doc-shuffle --update-ema-every 0 --swa-last-epochs 0 ^
  --weight-decay 0.4 ^
  --input_bin fineweb_data/fineweb_train.pt ^
  --input_val_bin fineweb_data/fineweb_val.pt ^
  --total-batch-size 16384
