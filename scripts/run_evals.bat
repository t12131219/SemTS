@echo off
REM SemTS full evaluation chain (run AFTER training): losses analysis + forecasting(ETT x4)
REM + zero-shot + anomaly(PSM) + classification(ArrowHead) + semantic probes.
REM Each step prints to console and appends to experiments\evals_log.txt.
REM Usage (from project root): run_evals.bat <checkpoint_path>
REM NOTE: keep this file pure ASCII (cmd parses .bat as GBK on Chinese Windows).
setlocal EnableExtensions
cd /d "%~dp0.."
if "%1"=="" (
    echo Usage: run_evals.bat experiments\pretrain_ettm1\ckpt_ep9.pt
    exit /b 1
)
set "CKPT=%~f1"
set "PY=E:\anaconda3\envs\SemTS\python.exe"
set "LOG=experiments\evals_log.txt"
> "%LOG%" echo.
set "TMPLOG=%TEMP%\semts_eval_%RANDOM%.log"

call :run analyze_losses scripts\analyze_losses.py --log experiments\pretrain_ettm1\losses.csv || exit /b 1

for %%D in (ETTm1 ETTh1 ETTh2 ETTm2) do (
    call :run finetune_%%D scripts\finetune_forecast.py --ckpt "%CKPT%" --data data\%%D.csv --horizons "96,192,336,720" || exit /b 1
)

call :run zeroshot scripts\eval_zeroshot.py --ckpt "%CKPT%" --data_src data\ETTm1.csv --data_tgt data\ETTh1.csv --horizons "96,336" || exit /b 1
call :run anomaly_PSM scripts\eval_anomaly.py --ckpt "%CKPT%" --data data\PSM\data.csv --labels data\PSM\label.npy --L 336 || exit /b 1
call :run classify_ArrowHead scripts\finetune_classify.py --ckpt "%CKPT%" --data_dir data\ArrowHead || exit /b 1

call :run probe_s1 scripts\probes.py --ckpt "%CKPT%" --probe s1_shuffle --data data\ETTm1.csv
call :run probe_s2 scripts\probes.py --ckpt "%CKPT%" --probe s2_pattern
call :run probe_s3 scripts\probes.py --ckpt "%CKPT%" --probe s3_lag
call :run probe_s4 scripts\probes.py --ckpt "%CKPT%" --probe s4_quality

echo ===== EVALS COMPLETE =====
echo Log: %CD%\%LOG%
del "%TMPLOG%" 2>nul
exit /b 0

:run
REM usage: call :run <step_name> <python_args...>
REM guard uses %~2 (quotes stripped) so quoted args like "96,192,336,720" do not break IF parsing;
REM ARGS keeps the raw %2 (quotes preserved) so commas stay grouped when expanded.
echo ===== STEP: %1 =====
set "STEPNAME=%1"
set "ARGS="
:run_loop
if "%~2"=="" goto run_exec
set "ARGS=%ARGS% %2"
shift /2
goto run_loop
:run_exec
"%PY%" %ARGS% > "%TMPLOG%" 2>&1
set "RC=%errorlevel%"
type "%TMPLOG%"
type "%TMPLOG%" >> "%LOG%"
if not "%RC%"=="0" (
    echo [FAIL] STEP %STEPNAME%
    exit /b 1
)
exit /b 0
