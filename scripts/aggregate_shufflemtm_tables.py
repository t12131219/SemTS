"""汇总 ShuffleMTM 复现结果：论文数字 + ShuffleMTM 官方实测 + SemTS，产出四张对比表。

输入（3090 结果回传后）：
  docs/shufflemtm_paper_numbers.json                论文数字（parse_shufflemtm_paper_tables.py 生成）
  third_party/ShuffleMTM/outputs/test_results/<d>/*.txt   官方 ShuffleMTM 预测（mse/mae）
  experiments/semts_tables.log                      SemTS 预测（H= 96: MSE=...）
  experiments/shufflemtm_classify/{AD,PTB}_results.json  官方 ShuffleMTM 分类（四指标 × 比例）
  experiments/shufflemtm_classify.log               SemTS 分类（test acc/prec/rec/f1）

输出：docs/shufflemtm_tables_results.md（论文同款四张表）
用法: python scripts/aggregate_shufflemtm_tables.py
"""
import glob
import json
import os
import re

PAPER = os.path.join("docs", "shufflemtm_paper_numbers.json")
SMTM_DIR = os.path.join("third_party", "ShuffleMTM", "outputs", "test_results")
SEMTS_LOG = os.path.join("experiments", "semts_tables.log")
CLS_DIR = os.path.join("experiments", "shufflemtm_classify")
CLS_LOG = os.path.join("experiments", "shufflemtm_classify.log")
OUT = os.path.join("docs", "shufflemtm_tables_results.md")

T1_MODELS = ["ShuffleMTM", "TimeSiam", "PITS", "PatchTST", "SimMTM",
             "iTransformer", "Crossformer", "CrossGNN", "MTGNN", "PatchTST_sup", "DLinear"]
T2_MODELS = ["ShuffleMTM", "TimeSiam", "PITS", "PatchTST", "SimMTM"]
T3_MODELS = ["ShuffleMTM", "TimeSiam", "PITS", "PatchTST", "SimMTM", "COMET"]
HORIZONS = ["96", "192", "336", "720"]


def load_paper():
    return json.load(open(PAPER, encoding="utf-8"))


def load_smtm_forecast(suffix=""):
    """{dataset: {horizon: [mse, mae]}}，取每个 setting 文件最后一行（最佳 epoch 结果）。
    suffix="" 读域内结果（*.txt 排除 *_cross）；suffix="_cross" 读跨域结果。"""
    out = {}
    pat = f"*{suffix}.txt" if suffix else "*.txt"
    for d in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"):
        res = {}
        for f in glob.glob(os.path.join(SMTM_DIR, d, pat)):
            if not suffix and f.endswith("_cross.txt"):
                continue
            for line in open(f, encoding="utf-8", errors="replace"):
                if not line.strip():
                    continue
                m = re.search(r"(\d+) -> (\d+) \| ([\d.]+), ([\d.]+)", line)
                if m:
                    # exp.test 按视野追加写；同一视野取最后一次出现（最新结果）
                    res[m.group(2)] = [float(m.group(3)), float(m.group(4))]
        out[d] = res
    return out


def load_semts_crossdomain():
    """{pair: [mse_avg, mae_avg]}，从 semts_crossdomain.log 抓取（4 视野平均）。"""
    out, cur = {}, None
    p = os.path.join("experiments", "semts_crossdomain.log")
    if not os.path.exists(p):
        return {}
    acc = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"===== cross (\w+)->(\w+) =====", line)
        if m:
            cur = f"{m.group(1)}->{m.group(2)}"
            acc[cur] = []
        m = re.search(r"H=\s*(\d+): MSE=([\d.]+) MAE=([\d.]+)", line)
        if m and cur:
            acc[cur].append([float(m.group(2)), float(m.group(3))])
    for k, v in acc.items():
        if len(v) == 4:
            out[k] = [sum(x[0] for x in v) / 4, sum(x[1] for x in v) / 4]
    return out


def load_semts_forecast():
    """{dataset: {horizon: [mse, mae]}}，从 semts_tables.log 抓取。"""
    out, cur = {}, None
    if not os.path.exists(SEMTS_LOG):
        return {}
    for line in open(SEMTS_LOG, encoding="utf-8", errors="replace"):
        m = re.search(r"===== eval (\w+) =====", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
        m = re.search(r"H=\s*(\d+): MSE=([\d.]+) MAE=([\d.]+)", line)
        if m and cur:
            out[cur][m.group(1)] = [float(m.group(2)), float(m.group(3))]
    return out


def load_smtm_classify():
    out = {}
    for name in ("AD", "PTB"):
        p = os.path.join(CLS_DIR, f"{name}_results.json")
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))
    return out


def load_semts_classify():
    """{dataset: {fraction: {acc, prec, rec, f1}}}。
    优先读 experiments/semts_classify_probe/{AD,PTB}.json（run_semts_classify_probe.py 产出），
    缺失时回退解析 classify 日志。"""
    out = {}
    for name in ("AD", "PTB"):
        p = os.path.join("experiments", "semts_classify_probe", f"{name}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))
    if out:
        return out
    # 回退：从 classify 日志解析
    cur, frac = None, None
    if not os.path.exists(CLS_LOG):
        return {}
    for line in open(CLS_LOG, encoding="utf-8", errors="replace"):
        m = re.search(r"SemTS\s*探针[:：]\s*(AD|PTB)", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
        m = re.search(r"--fraction ([\d.]+)", line)
        if m:
            frac = f"{float(m.group(1)):g}"
        m = re.search(r"test acc=([\d.]+) prec=([\d.]+) rec=([\d.]+) f1=([\d.]+)", line)
        if m and cur and frac:
            out[cur][frac] = {k: float(v) for k, v in
                              zip(("acc", "prec", "rec", "f1"), m.groups())}
    return out


def fmt(v, nd=3):
    return f"{v:.{nd}f}"


def main():
    paper = load_paper()
    smtm_f = load_smtm_forecast()
    semts_f = load_semts_forecast()
    smtm_c = load_smtm_classify()
    semts_c = load_semts_classify()

    L = []
    a = L.append
    a("# ShuffleMTM (ICLR 2025) 表格复现结果汇总")
    a("")
    a("> 生成方式：`python scripts/aggregate_shufflemtm_tables.py`。")
    a("> 列说明：**Paper** = 论文报告数字；**Ours** = 本次官方代码实测；"
      "**SemTS** = 我们的模型（S1+S3+S4 主配置 + 统一协议）。其余基线列引用论文数字。")
    a("")

    # ===== Table 1 =====
    a("## Table 1: 域内长时预测（MSE，8 数据集 × 4 视野）")
    a("")
    for ds, rows in paper["table1_in_domain_forecast"].items():
        a(f"### {ds}")
        a("")
        head = "| H | " + " | ".join(m for m in T1_MODELS) + " | SemTS |"
        a(head)
        a("|---|" + "---|" * (len(T1_MODELS) + 1))
        for h in HORIZONS + ["Avg"]:
            cells = []
            for m in T1_MODELS:
                if m == "ShuffleMTM":
                    if ds in smtm_f and h in smtm_f[ds]:
                        cells.append(f"{fmt(paper['table1_in_domain_forecast'][ds][h][m][0])} / **{fmt(smtm_f[ds][h][0])}**")
                    else:
                        cells.append(fmt(paper["table1_in_domain_forecast"][ds][h][m][0]))
                else:
                    cells.append(fmt(paper["table1_in_domain_forecast"][ds][h][m][0]))
            if ds in semts_f and h in semts_f[ds]:
                cells.append(f"**{fmt(semts_f[ds][h][0])}**")
            else:
                cells.append("-")
            a("| " + h + " | " + " | ".join(cells) + " |")
        a("")
    a("")

    # ===== Table 2 =====
    smtm_cross = load_smtm_forecast("_cross")
    semts_cross = load_semts_crossdomain()
    a("## Table 2: 跨域预测（MSE，源域→目标域，4 视野平均）")
    a("")
    head = "| Pair | " + " | ".join(T2_MODELS) + " | SemTS |"
    a(head)
    a("|---|" + "---|" * (len(T2_MODELS) + 1))
    for pair, rows in paper["table2_cross_domain"].items():
        cells = []
        for m in T2_MODELS:
            if m == "ShuffleMTM":
                tgt = pair.split("->")[1]
                # 官方实测：跨域结果（_cross 文件，目标域 4 视野平均）
                if tgt in smtm_cross and all(h in smtm_cross[tgt] for h in HORIZONS):
                    avg = sum(smtm_cross[tgt][h][0] for h in HORIZONS) / 4
                    cells.append(f"{fmt(rows[m][0])} / **{fmt(avg)}**")
                else:
                    cells.append(fmt(rows[m][0]))
            else:
                cells.append(fmt(rows[m][0]))
        if pair in semts_cross:
            cells.append(f"**{fmt(semts_cross[pair][0])}**")
        else:
            cells.append("-")
        a("| " + pair.replace("->", "→") + " | " + " | ".join(cells) + " |")
    a("")

    # ===== Table 3 =====
    a("## Table 3: 域内分类（Accuracy / Precision / Recall / F1）")
    a("")
    for ds, rows in paper["table3_classification"].items():
        a(f"### {ds}")
        a("")
        a("| Model | Acc (Paper/Ours) | Prec | Rec | F1 |")
        a("|---|---|---|---|---|")
        for m in T3_MODELS:
            r = rows[m]
            if m == "ShuffleMTM" and ds in smtm_c and "1.0" in smtm_c[ds]:
                o = smtm_c[ds]["1.0"]
                a(f"| {m} | {fmt(r['acc'])} / **{fmt(o['acc']*100 if o['acc']<=1 else o['acc'])}** | "
                  f"{fmt(r['prec'])} | {fmt(r['rec'])} | {fmt(r['f1'])} |")
            else:
                a(f"| {m} | {fmt(r['acc'])} | {fmt(r['prec'])} | {fmt(r['rec'])} | {fmt(r['f1'])} |")
        if ds in semts_c and "1.0" in semts_c[ds]:
            o = semts_c[ds]["1.0"]
            a(f"| **SemTS** | **{fmt(o['acc']*100 if o['acc']<=1 else o['acc'])}** | "
              f"**{fmt(o['prec']*100 if o['prec']<=1 else o['prec'])}** | "
              f"**{fmt(o['rec']*100 if o['rec']<=1 else o['rec'])}** | "
              f"**{fmt(o['f1']*100 if o['f1']<=1 else o['f1'])}** |")
        a("")
    a("")

    # ===== Table 4 =====
    a("## Table 4: 少量标签分类（Accuracy，标签比例 20% / 10% / 5%）")
    a("")
    for frac, rows in paper["table4_limited_labels"].items():
        a(f"### {frac}")
        a("")
        a("| Model | AD (Paper/Ours) | PTB (Paper/Ours) |")
        a("|---|---|---|")
        for m in T3_MODELS:
            r = rows[m]
            ad = r["AD"]["acc"]
            ptb = r["PTB"]["acc"]
            if m == "ShuffleMTM":
                fs = f"{frac[0]}{frac[1:]}"
                fk = {"20%": "0.2", "10%": "0.1", "5%": "0.05"}[frac]
                oad = smtm_c.get("AD", {}).get(fk, {}).get("acc", None)
                optb = smtm_c.get("PTB", {}).get(fk, {}).get("acc", None)
                cad = f"{fmt(ad)} / **{fmt(oad*100 if oad is not None and oad<=1 else oad)}**" if oad is not None else fmt(ad)
                cpt = f"{fmt(ptb)} / **{fmt(optb*100 if optb is not None and optb<=1 else optb)}**" if optb is not None else fmt(ptb)
                a(f"| {m} | {cad} | {cpt} |")
            else:
                a(f"| {m} | {fmt(ad)} | {fmt(ptb)} |")
        fk = {"20%": "0.2", "10%": "0.1", "5%": "0.05"}[frac]
        sad = semts_c.get("AD", {}).get(fk, {}).get("acc", None)
        spt = semts_c.get("PTB", {}).get(fk, {}).get("acc", None)
        if sad or spt:
            a(f"| **SemTS** | **{fmt(sad*100 if sad is not None and sad<=1 else sad)}** "
              f"| **{fmt(spt*100 if spt is not None and spt<=1 else spt)}** |")
        a("")

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"aggregated -> {OUT}")


if __name__ == "__main__":
    main()
