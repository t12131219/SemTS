@echo off
REM SemTS one-click pipeline (run locally in a terminal).
REM Flow: env check -> smoke test -> download ETTm1 -> CPU pretrain smoke
REM -> [CUDA full pretrain -> forecasting eval].
REM Each step prints to console and appends to experiments\pipeline.log; fails fast.
REM NOTE: keep this file pure ASCII (cmd parses .bat as GBK on Chinese Windows).
setlocal EnableExtensions
cd /d "%~dp0.."

call conda activate semts
if errorlevel 1 (
    echo [FAIL] conda activate semts failed
    exit /b 1
)
where python
python -V

mkdir experiments 2>nul
set "LOG=experiments\pipeline.log"
> "%LOG%" echo.
set "TMPLOG=%TEMP%\semts_step_%RANDOM%.log"

call :run check_env -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit /b 1
call :run smoke_test scripts\smoke_test.py || exit /b 1
call :run download_data scripts\download_data.py --out data --datasets ETTm1 || exit /b 1
call :run pretrain_cpu_smoke scripts\pretrain.py --data data\ETTm1.csv --max_batches 5 --batch_size 8 || exit /b 1

python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo [WARN] CUDA not available, skipping full pretrain and eval (CPU smoke only)
    goto :done
)

call :run pretrain_cuda scripts\pretrain.py --data data\ETTm1.csv --epochs 10 --batch_size 128 || exit /b 1
call :run finetune_forecast scripts\finetune_forecast.py --ckpt experiments\pretrain_ettm1\ckpt_ep9.pt --data data\ETTm1.csv || exit /b 1

:done
echo ===== PIPELINE COMPLETE =====
echo Log: %CD%\%LOG%
del "%TMPLOG%" 2>nul
exit /b 0

:run
REM usage: call :run <step_name> <python_args...>
REM guard uses %~2 (quotes stripped) so quoted args do not break IF parsing.
echo ===== STEP: %1 =====
set "STEPNAME=%1"
set "ARGS="
:run_loop
if "%~2"=="" goto run_exec
set "ARGS=%ARGS% %2"
shift /2
goto run_loop
:run_exec
python %ARGS% > "%TMPLOG%" 2>&1
set "RC=%errorlevel%"
type "%TMPLOG%"
type "%TMPLOG%" >> "%LOG%"
if not "%RC%"=="0" (
    echo [FAIL] STEP %STEPNAME%
    exit /b 1
)
exit /b 0
