"""把 ShuffleMTM 复现数据渲染为论文版式四张表（SemTS / ShuffleMTM 作为一等模型列出）。

版式与论文一致：
  Table 1：行 = 数据集×视野（含 Avg），列 = 模型（MSE/MAE 成对）
  Table 2：行 = 源→目标对，列 = 模型（4 视野平均 MSE）
  Table 3：AD/PTB 两个子表，行 = 模型，列 = Acc/Prec/Rec/F1
  Table 4：行 = 模型，列 = 数据集 × 标签比例（20/10/5%）

输入（本地已齐）：
  docs/shufflemtm_paper_numbers.json           论文数字
  third_party/ShuffleMTM/outputs/test_results/ 官方实测（域内 + _cross）
  experiments/semts_tables.log / semts_crossdomain.log
  experiments/shufflemtm_classify/{AD,PTB}_results.json（官方分类实测）
  experiments/semts_classify_probe/{AD,PTB}.json（SemTS 探针）
输出：docs/shufflemtm_tables_paper_style.md
用法: python scripts/render_paper_style_tables.py
"""
import glob
import json
import os
import re

PAPER = os.path.join("docs", "shufflemtm_paper_numbers.json")
SMTM_DIR = os.path.join("third_party", "ShuffleMTM", "outputs", "test_results")
SEMTS_LOG = os.path.join("experiments", "semts_tables.log")
SEMTS_CROSS = os.path.join("experiments", "semts_crossdomain.log")
CLS_DIR = os.path.join("experiments", "shufflemtm_classify")
PROBE_DIR = os.path.join("experiments", "semts_classify_probe")
OUT = os.path.join("docs", "shufflemtm_tables_paper_style.md")

HORIZONS = ["96", "192", "336", "720"]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
PAIRS = ["ETTh1->ETTh2", "ETTh2->ETTh1", "ETTh1->ETTm1", "ETTh2->ETTm1",
         "ETTm2->ETTm1", "ETTm2->ETTh1", "Weather->ETTh1", "Weather->ETTm1"]
T1_OTHER = ["TimeSiam", "PITS", "PatchTST", "SimMTM", "iTransformer",
            "Crossformer", "CrossGNN", "MTGNN", "PatchTST_sup", "DLinear"]
T23_OTHER = ["TimeSiam", "PITS", "PatchTST", "SimMTM", "COMET"]


def load_paper():
    return json.load(open(PAPER, encoding="utf-8"))


def load_smtm(suffix=""):
    pat = f"*{suffix}.txt" if suffix else "*.txt"
    out = {}
    for d in DATASETS:
        res = {}
        for f in glob.glob(os.path.join(SMTM_DIR, d, pat)):
            if not suffix and f.endswith("_cross.txt"):
                continue
            for line in open(f, encoding="utf-8", errors="replace"):
                m = re.search(r"(\d+) -> (\d+) \| ([\d.]+), ([\d.]+)", line)
                if m:
                    res[m.group(2)] = [float(m.group(3)), float(m.group(4))]
        out[d] = res
    return out


def load_semts_forecast():
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


def load_semts_cross():
    out, cur, acc = {}, None, {}
    if not os.path.exists(SEMTS_CROSS):
        return {}
    for line in open(SEMTS_CROSS, encoding="utf-8", errors="replace"):
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


def load_cls(directory):
    out = {}
    for name in ("AD", "PTB"):
        p = os.path.join(directory, f"{name}.json") if os.path.isdir(directory) else None
        if p and os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))
    return out


def pair(v, nd=3):
    if v is None:
        return "-"
    return f"{v[0]:.{nd}f}/{v[1]:.{nd}f}"


def pct(v):
    if v is None:
        return "-"
    return f"{v * 100:.3f}" if v <= 1 else f"{v:.3f}"


def main():
    paper = load_paper()
    smtm = load_smtm()
    smtm_cross = load_smtm("_cross")
    semts = load_semts_forecast()
    semts_cross = load_semts_cross()
    smtm_cls = load_cls(CLS_DIR)
    semts_cls = load_cls(PROBE_DIR)

    L = []
    a = L.append
    a("# ShuffleMTM 复现结果（论文版式，SemTS 与 ShuffleMTM 为一等模型）")
    a("")
    a("> 每格为 MSE/MAE（Table 3/4 为百分比指标）。ShuffleMTM(Ours) 为本次官方代码单次实测；")
    a("> SemTS 为 S1+S3+S4 主配置（10 epoch + 冻结线性探针/岭回归）。其余基线引用论文数字。")
    a("")

    # ===== Table 1 =====
    a("## Table 1: In-domain forecasting (MSE/MAE)")
    a("")
    cols = ["Dataset", "H", "ShuffleMTM", "ShuffleMTM(Ours)", "SemTS"] + T1_OTHER
    a("| " + " | ".join(cols) + " |")
    a("|" + "---|" * len(cols))
    t1 = paper["table1_in_domain_forecast"]
    for ds in DATASETS:
        for h in HORIZONS + ["Avg"]:
            row = [ds if h == "96" else "", h]
            row.append(pair(t1[ds][h]["ShuffleMTM"]))
            ours = None
            if h == "Avg" and all(x in smtm.get(ds, {}) for x in HORIZONS):
                ours = [sum(smtm[ds][x][0] for x in HORIZONS) / 4,
                        sum(smtm[ds][x][1] for x in HORIZONS) / 4]
            elif h in smtm.get(ds, {}):
                ours = smtm[ds][h]
            row.append(pair(ours))
            st = None
            if h == "Avg" and all(x in semts.get(ds, {}) for x in HORIZONS):
                st = [sum(semts[ds][x][0] for x in HORIZONS) / 4,
                      sum(semts[ds][x][1] for x in HORIZONS) / 4]
            elif h in semts.get(ds, {}):
                st = semts[ds][h]
            row.append(pair(st))
            for m in T1_OTHER:
                row.append(pair(t1[ds][h][m]))
            a("| " + " | ".join(row) + " |")
        a("")
    a("")

    # ===== Table 2 =====
    a("## Table 2: Cross-domain forecasting (MSE/MAE, avg over 4 horizons)")
    a("")
    cols = ["Pair", "ShuffleMTM", "ShuffleMTM(Ours)", "SemTS", "TimeSiam", "PITS", "PatchTST", "SimMTM"]
    a("| " + " | ".join(cols) + " |")
    a("|" + "---|" * len(cols))
    t2 = paper["table2_cross_domain"]
    for p in PAIRS:
        tgt = p.split("->")[1]
        ours = None
        if tgt in smtm_cross and all(h in smtm_cross[tgt] for h in HORIZONS):
            ours = [sum(smtm_cross[tgt][h][0] for h in HORIZONS) / 4,
                    sum(smtm_cross[tgt][h][1] for h in HORIZONS) / 4]
        row = [p.replace("->", "→"), pair(t2[p]["ShuffleMTM"]), pair(ours),
               pair(semts_cross.get(p))]
        for m in ["TimeSiam", "PITS", "PatchTST", "SimMTM"]:
            row.append(pair(t2[p][m]))
        a("| " + " | ".join(row) + " |")
    a("")

    # ===== Table 3 =====
    for ds in ("AD", "PTB"):
        a(f"## Table 3: In-domain classification — {ds} (Acc/Prec/Rec/F1, %)")
        a("")
        a("| Model | Acc | Prec | Rec | F1 |")
        a("|---|---|---|---|---|")
        t3 = paper["table3_classification"][ds]
        o = smtm_cls.get(ds, {}).get("1.0", {})
        a("| ShuffleMTM | " + " | ".join(f"{t3['ShuffleMTM'][k]:.3f}" for k in ("acc", "prec", "rec", "f1")) + " |")
        a("| ShuffleMTM(Ours) | " + " | ".join(pct(o.get(k)) for k in ("acc", "prec", "rec", "f1")) + " |")
        s = semts_cls.get(ds, {}).get("1.0", {})
        a("| **SemTS** | " + " | ".join(pct(s.get(k)) for k in ("acc", "prec", "rec", "f1")) + " |")
        for m in T23_OTHER:
            a("| " + m + " | " + " | ".join(f"{t3[m][k]:.3f}" for k in ("acc", "prec", "rec", "f1")) + " |")
        a("")
    a("")

    # ===== Table 4 =====
    a("## Table 4: Limited labeled data classification (Acc, %)")
    a("")
    a("| Model | AD 20% | AD 10% | AD 5% | PTB 20% | PTB 10% | PTB 5% |")
    a("|---|---|---|---|---|---|---|")
    t4 = paper["table4_limited_labels"]
    fracs = {"0.2": "20%", "0.1": "10%", "0.05": "5%"}
    row = ["ShuffleMTM"]
    for ds in ("AD", "PTB"):
        for fk in ("0.2", "0.1", "0.05"):
            row.append(f"{t4[fracs[fk]]['ShuffleMTM'][ds]['acc']:.3f}")
    a("| " + " | ".join(row) + " |")
    row = ["ShuffleMTM(Ours)"]
    for ds in ("AD", "PTB"):
        for fk in ("0.2", "0.1", "0.05"):
            row.append(pct(smtm_cls.get(ds, {}).get(fk, {}).get("acc")))
    a("| " + " | ".join(row) + " |")
    row = ["**SemTS**"]
    for ds in ("AD", "PTB"):
        for fk in ("0.2", "0.1", "0.05"):
            row.append(pct(semts_cls.get(ds, {}).get(fk, {}).get("acc")))
    a("| " + " | ".join(row) + " |")
    for m in T23_OTHER:
        row = [m]
        for ds in ("AD", "PTB"):
            for fk in ("0.2", "0.1", "0.05"):
                row.append(f"{t4[fracs[fk]][m][ds]['acc']:.3f}")
        a("| " + " | ".join(row) + " |")
    a("")

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"rendered -> {OUT}")


if __name__ == "__main__":
    main()
