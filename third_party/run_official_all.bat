@echo off
REM Official-implementation comparison: TS2Vec + CoST + SimMTM on ETTm1.
REM Each: official pretraining (same data/splits/step-budget as SemTS) -> feature export
REM -> unified ridge protocol (scripts\finetune_from_features.py).
REM Usage (from SemTS project root): third_party\run_official_all.bat <n_iters=3240>
REM NOTE: keep this file pure ASCII (cmd parses .bat as GBK on Chinese Windows).
setlocal EnableExtensions
cd /d "%~dp0.."
set "NIT=3240"
if not "%1"=="" set "NIT=%1"
call conda activate semts
if errorlevel 1 ( echo [FAIL] conda activate semts failed & exit /b 1 )

echo ===== official ts2vec: pretrain + export =====
python third_party\run_official_ts2vec.py --data data\ETTm1.csv --n_iters %NIT% || exit /b 1
python scripts\finetune_from_features.py --feats experiments\official_ts2vec\feats --data data\ETTm1.csv || exit /b 1

echo ===== official CoST: pretrain + export =====
python third_party\run_official_cost.py --data data\ETTm1.csv --n_iters %NIT% || exit /b 1
python scripts\finetune_from_features.py --feats experiments\official_cost\feats --data data\ETTm1.csv || exit /b 1

REM NOTE: official SimMTM excluded from the official-implementation comparison by user decision.
REM (Its Autoformer-style framework hangs on Windows dataloader spawn; the unified-backbone
REM  SimMTM-style baseline remains available in scripts\pretrain_baselines.py for method comparison.)

echo ===== official ShuffleMTM: pretrain (their framework, official ETT config) =====
pushd third_party\ShuffleMTM
python run.py --task_name pretrain --data ETTm1 --gpu 0 --seed 2024 --e_layers 3 --n_heads 16 --d_model 32 --d_ff 128 --patch_len 6 --stride 6 --pretrain_learning_rate 5e-4 --seq_len 336 --root_path ..\..\data --data_path ETTm1.csv --features M --train_epochs 10 --batch_size 64 --num_workers 0 || ( popd & exit /b 1 )
popd

echo ===== official ShuffleMTM: export features + unified ridge =====
python third_party\run_official_shufflemtm.py --export --data data\ETTm1.csv || exit /b 1
python scripts\finetune_from_features.py --feats experiments\official_shufflemtm\feats --data data\ETTm1.csv || exit /b 1

echo ===== OFFICIAL COMPARISON COMPLETE =====
