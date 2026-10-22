@echo off
REM Unified-backbone baselines for RTX 3090 24GB (batch 256 + TF32, sequential).
REM Each baseline: pretrain -> ETTm1 linear-eval (ridge) with the SAME protocol.
REM Usage: run_baselines_3090.bat <epochs=10>
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

for %%B in (ts2vec cost simmtm tf_c) do (
    echo ===== baseline: %%B (pretrain) =====
    python scripts\pretrain_baselines.py --baseline %%B --data data\ETTm1.csv --epochs %EP% --batch_size 256 --tf32 || exit /b 1
    echo ===== baseline: %%B (finetune eval) =====
    python scripts\finetune_forecast.py --ckpt experiments\pretrain_%%B\ckpt_ep%LAST%.pt --data data\ETTm1.csv --horizons "96,336" || exit /b 1
)

echo ===== BASELINES COMPLETE =====
