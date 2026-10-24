#!/usr/bin/env bash
# 官方实现对照：克隆官方仓库并应用统一协议补丁。
# 在 SemTS/third_party 目录下运行：bash setup_repos.sh
# 依赖安装：pip install -r third_party/requirements_official.txt
set -e
cd "$(dirname "$0")"

[ -d ts2vec ]    || git clone --depth 1 https://github.com/zhihanyue/ts2vec.git
[ -d CoST ]      || git clone --depth 1 https://github.com/salesforce/CoST.git
[ -d ShuffleMTM ] || git clone --depth 1 https://github.com/cikm2025shufflemtm/ShuffleMTM.git

# 统一协议补丁：数据切分 0.7/0.2/0.1 → 0.6/0.2/0.2（与 SemTS 一致）
sed -i 's/int(len(df_raw) \* 0\.7)/int(len(df_raw) * 0.6)/g' ShuffleMTM/data_provider/data_loader.py
sed -i 's/int(len(df_raw) \* 0\.1)/int(len(df_raw) * 0.2)/g' ShuffleMTM/data_provider/data_loader.py
# NumPy 2.0 兼容补丁（官方 EarlyStopping 使用已移除的 np.Inf，微调阶段会崩溃）
sed -i 's/self\.val_loss_min = np\.Inf/self.val_loss_min = np.inf/' ShuffleMTM/utils/tools.py
# 硬件加速补丁（协议无损）：TF32 + cuDNN 自动调优 + pin_memory
python - <<'EOF'
import re
p = 'ShuffleMTM/run.py'
s = open(p).read()
if 'cudnn.benchmark' not in s:
    s = s.replace("os.environ['CUDA_LAUNCH_BLOCKING'] = '0'",
                  "os.environ['CUDA_LAUNCH_BLOCKING'] = '0'\n"
                  "if torch.cuda.is_available():\n"
                  "    torch.backends.cudnn.benchmark = True\n"
                  "    torch.backends.cuda.matmul.allow_tf32 = True")
    open(p, 'w').write(s)
p = 'ShuffleMTM/data_provider/data_factory.py'
s = open(p).read()
s = s.replace('drop_last=drop_last)', 'drop_last=drop_last,\n        pin_memory=True)')
open(p, 'w').write(s)
# 结果文件后缀补丁：跨域结果写 _cross.txt，与域内结果分离（避免追加混淆）
p = 'ShuffleMTM/run.py'
s = open(p).read()
if 'result_suffix' not in s:
    s = s.replace("parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')",
                  "parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')\n"
                  "parser.add_argument('--result_suffix', type=str, default='', help='suffix for test result txt file')")
    open(p, 'w').write(s)
p = 'ShuffleMTM/exp/exp_shufflemtm.py'
s = open(p).read()
s = s.replace('f = open(folder_path + "{}.txt".format(setting), \'a\')',
              'f = open(folder_path + "{}{}.txt".format(setting, self.args.result_suffix), \'a\')')
open(p, 'w').write(s)
print('hardware acceleration patches applied')
EOF

echo "official repos ready"
echo "next: bash third_party/run_official_all.sh   (见 docs/setup.md 的执行顺序)"
