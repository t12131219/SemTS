@echo off
REM Ablation experiments for RTX 3090 24GB (batch 256 + TF32, sequential).
REM Usage: run_ablations_3090.bat <epochs=10>
REM NOTE: keep this file pure ASCII (cmd parses .bat as GBK on Chinese Windows).
setlocal EnableExtensions
cd /d "%~dp0.."
set "EP=10"
if not "%1"=="" set "EP=%1"
set /a "LAST=%EP%-1"
call conda activate semts
if errorlevel 1 (
    echo [FAIL] conda activate semts failed
    exit /b 1
)

echo ===== ablation: -S3 =====
python scripts\pretrain.py --data data\ETTm1.csv --epochs %EP% --batch_size 256 --tf32 --out experiments\ablation_w3 --w3 0 || exit /b 1
python scripts\finetune_forecast.py --ckpt experiments\ablation_w3\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1

echo ===== ablation: -S4 =====
python scripts\pretrain.py --data data\ETTm1.csv --epochs %EP% --batch_size 256 --tf32 --out experiments\ablation_w4 --w4 0 || exit /b 1
python scripts\finetune_forecast.py --ckpt experiments\ablation_w4\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1

echo ===== ablation: S1+S2 only =====
python scripts\pretrain.py --data data\ETTm1.csv --epochs %EP% --batch_size 256 --tf32 --out experiments\ablation_s12 --w3 0 --w4 0 || exit /b 1
python scripts\finetune_forecast.py --ckpt experiments\ablation_s12\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1

echo ===== ABLATIONS COMPLETE =====
