@echo off
REM Ablation experiments (run AFTER main training, sequential): -S3 / -S4 / S1+S2 only.
REM Usage: run_ablations.bat <epochs=3>
REM NOTE: keep this file pure ASCII (cmd parses .bat as GBK on Chinese Windows).
setlocal EnableExtensions
cd /d "%~dp0.."
set "EP=3"
if not "%1"=="" set "EP=%1"
set /a "LAST=%EP%-1"
set "PY=E:\anaconda3\envs\SemTS\python.exe"

echo ===== ablation: -S3 =====
"%PY%" scripts\pretrain.py --data data\ETTm1.csv --epochs %EP% --batch_size 128 --out experiments\ablation_w3 --w3 0 || exit /b 1
"%PY%" scripts\finetune_forecast.py --ckpt experiments\ablation_w3\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1

echo ===== ablation: -S4 =====
"%PY%" scripts\pretrain.py --data data\ETTm1.csv --epochs %EP% --batch_size 128 --out experiments\ablation_w4 --w4 0 || exit /b 1
"%PY%" scripts\finetune_forecast.py --ckpt experiments\ablation_w4\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1

echo ===== ablation: S1+S2 only =====
"%PY%" scripts\pretrain.py --data data\ETTm1.csv --epochs %EP% --batch_size 128 --out experiments\ablation_s12 --w3 0 --w4 0 || exit /b 1
"%PY%" scripts\finetune_forecast.py --ckpt experiments\ablation_s12\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1

echo ===== ABLATIONS COMPLETE =====
