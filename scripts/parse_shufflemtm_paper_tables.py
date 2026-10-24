"""把 _shufflemtm_text.txt 中论文四张主结果表解析为结构化 JSON。

输出 docs/shufflemtm_paper_numbers.json：
  table1_in_domain_forecast:  {dataset: {horizon: {model: [mse, mae]}}}
  table2_cross_domain:        {pair: {model: [mse, mae]}}
  table3_classification:      {dataset: {model: {acc, prec, rec, f1}}}
  table4_limited_labels:      {fraction: {dataset: {model: {acc, prec, rec, f1}}}}

用法: python scripts/parse_shufflemtm_paper_tables.py
"""
import json
import os
import re
import sys

TXT = os.path.join("..", "Atricles", "Time Series Forecastinng", "PreTrain", "_shufflemtm_text.txt")

MODELS_T1 = ["ShuffleMTM", "TimeSiam", "PITS", "PatchTST", "SimMTM",
             "iTransformer", "Crossformer", "CrossGNN", "MTGNN", "PatchTST_sup", "DLinear"]
MODELS_T2 = ["ShuffleMTM", "TimeSiam", "PITS", "PatchTST", "SimMTM"]
MODELS_T3 = ["ShuffleMTM", "TimeSiam", "PITS", "PatchTST", "SimMTM", "COMET"]


def nums(line):
    return [float(x) for x in re.findall(r"\d+\.\d+", line)]


def main():
    lines = open(TXT, encoding="utf-8").read().splitlines()
    out = {}

    # ===== Table 1（行 636-683: 8 数据集 × 5 行 × 22 值 = MSE/MAE × 11 模型）=====
    t1 = {}
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather",
               "Electricity", "Traffic"]:
        ds_lines = {}
        for i, l in enumerate(lines):
            if l.strip() == ds:
                rows = lines[i + 1:i + 6]
                break
        assert len(rows) == 5, ds
        horizons = ["96", "192", "336", "720", "Avg"]
        for h, row in zip(horizons, rows):
            v = nums(row)
            assert len(v) == 22, (ds, h, len(v))
            ds_lines[h] = {m: v[2 * j:2 * j + 2] for j, m in enumerate(MODELS_T1)}
        t1[ds] = ds_lines
    out["table1_in_domain_forecast"] = t1

    # ===== Table 2（行 688-695: 8 对 × 10 值）=====
    t2 = {}
    pairs = ["ETTh1->ETTh2", "ETTh2->ETTh1", "ETTh1->ETTm1", "ETTh2->ETTm1",
             "ETTm2->ETTm1", "ETTm2->ETTh1", "Weather->ETTh1", "Weather->ETTm1"]
    for i, l in enumerate(lines):
        if l.strip().startswith("ETTh1→ ETTh2"):
            block = lines[i:i + 8]
            break
    for p, row in zip(pairs, block):
        v = nums(row)
        assert len(v) == 10, (p, len(v))
        t2[p] = {m: v[2 * j:2 * j + 2] for j, m in enumerate(MODELS_T2)}
    out["table2_cross_domain"] = t2

    # ===== Table 3（行 699-713: 2 数据集 × 6 模型 × 4 指标）=====
    t3 = {}
    for i, l in enumerate(lines):
        if l.strip() == "AD" and i > 690:
            ad_block = lines[i + 1:i + 7]
            break
    for i, l in enumerate(lines):
        if l.strip() == "PTB" and i > 690:
            ptb_block = lines[i + 1:i + 7]
            break
    for ds, block in (("AD", ad_block), ("PTB", ptb_block)):
        d = {}
        for row in block:
            name = row.split()[0]
            v = nums(row)
            assert len(v) == 4, (ds, name, v)
            d[name] = dict(zip(["acc", "prec", "rec", "f1"], v))
        t3[ds] = d
    out["table3_classification"] = t3

    # ===== Table 4（固定行号：表头 20%=793、10%=800、5%=807，各取后 6 行，1-based）=====
    t4 = {}
    blocks = {"20%": (793, 799), "10%": (800, 806), "5%": (807, 813)}  # (表头, 末行)
    for frac, (hdr, end) in blocks.items():
        assert lines[hdr - 1].strip() == frac, (frac, lines[hdr - 1])
        d = {}
        for row in lines[hdr:end]:  # 0-based: hdr..end-1
            name = row.split()[0]
            # 提取的文本混入页面行号（小数 0.x~16.x），指标均为百分比（>50）——过滤小值
            v = [x for x in nums(row) if x > 50]
            assert len(v) == 8, (frac, name, v)
            d[name] = {"AD": dict(zip(["acc", "prec", "rec", "f1"], v[:4])),
                       "PTB": dict(zip(["acc", "prec", "rec", "f1"], v[4:]))}
        t4[frac] = d
    out["table4_limited_labels"] = t4

    dst = r"docs\shufflemtm_paper_numbers.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    n1 = sum(len(h) * 11 for h in t1.values())
    print(f"parsed: t1 {len(t1)} datasets x {n1} cells, t2 {len(t2)} pairs, "
          f"t3 {len(t3)} datasets, t4 {len(t4)} fractions -> {dst}")


if __name__ == "__main__":
    main()
