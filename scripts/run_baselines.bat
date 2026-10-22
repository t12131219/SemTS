@echo off
REM Unified-backbone baselines (run AFTER main training, sequential): ts2vec/cost/simmtm/tf_c.
REM Usage: run_baselines.bat <epochs=3>
REM NOTE: keep this file pure ASCII (cmd parses .bat as GBK on Chinese Windows).
setlocal EnableExtensions
cd /d "%~dp0.."
set "EP=3"
if not "%1"=="" set "EP=%1"
set "PY=E:\anaconda3\envs\SemTS\python.exe"

for %%B in (ts2vec cost simmtm tf_c) do (
    echo ===== baseline: %%B =====
    "%PY%" scripts\pretrain_baselines.py --baseline %%B --data data\ETTm1.csv --epochs %EP% --batch_size 128 || exit /b 1
)

echo ===== BASELINES COMPLETE =====
