# SemTS 一键流水线（DSH 会话执行层故障期间，由用户在本地终端运行）
# 流程：环境检查 → 冒烟测试 → 下载 ETTm1 → CPU 预训练冒烟 → [CUDA 正式预训练 → 预测评估]
# 每步输出实时打印并追加到 experiments\pipeline.log；任一步失败立即停止并报告。
# 脚本内部自动切到项目根目录，从任意位置启动均可。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -SkipCuda          # 只跑 CPU 冒烟
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Epochs 10 -BatchSize 128
param(
    [switch]$SkipCuda,
    [int]$Epochs = 10,
    [int]$BatchSize = 128
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Data = Join-Path $Root "data"
$Exp  = Join-Path $Root "experiments"
$Log  = Join-Path $Exp "pipeline.log"
New-Item -ItemType Directory -Force -Path $Exp | Out-Null
"" | Set-Content -Path $Log

function Step([string]$Name, [string[]]$ArgList) {
    Write-Host ("`n===== STEP: {0} =====" -f $Name)
    & conda run -n semts python @ArgList 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        throw ("STEP FAILED: {0} (exit {1})" -f $Name, $LASTEXITCODE)
    }
}

try {
    Push-Location $Root  # 保证所有脚本的相对路径（experiments/ 等）落在项目根目录
    Step "check_env" @("-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    Step "smoke_test" @((Join-Path $Root "scripts\smoke_test.py"))
    Step "download_data" @((Join-Path $Root "scripts\download_data.py"), "--out", $Data, "--datasets", "ETTm1")
    Step "pretrain_cpu_smoke" @((Join-Path $Root "scripts\pretrain.py"), "--data", (Join-Path $Data "ETTm1.csv"), "--max_batches", "5", "--batch_size", "8")

    if (-not $SkipCuda) {
        $cudaOut = & conda run -n semts python -c "import torch; print(torch.cuda.is_available())" 2>$null
        $cudaOk = ("$cudaOut").Trim()
        if ($cudaOk -ne "True") {
            throw "CUDA 不可用，无法进行正式预训练（如需纯 CPU 冒烟请加 -SkipCuda）"
        }
        Step "pretrain_cuda" @((Join-Path $Root "scripts\pretrain.py"), "--data", (Join-Path $Data "ETTm1.csv"), "--epochs", "$Epochs", "--batch_size", "$BatchSize")
        $ckpt = Join-Path $Exp ("pretrain_ettm1\ckpt_ep{0}.pt" -f ($Epochs - 1))
        Step "finetune_forecast" @((Join-Path $Root "scripts\finetune_forecast.py"), "--ckpt", $ckpt, "--data", (Join-Path $Data "ETTm1.csv"))
    }

    Pop-Location
    Write-Host "`n===== PIPELINE COMPLETE ====="
    Write-Host "日志: $Log"
}
catch {
    Pop-Location -ErrorAction SilentlyContinue
    Write-Host "`n===== PIPELINE ABORTED ====="
    Write-Host $_.Exception.Message
    $_ | Out-File -FilePath $Log -Append
    exit 1
}
