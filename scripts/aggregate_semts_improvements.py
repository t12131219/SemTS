"""汇总 SemTS 改进运动结果：解析 semts_improvement.log → 对比表。

输出 docs/semts_improvement_results.md：
  每数据集一张表：H × [ShuffleMTM(论文) | SemTS-10ep岭 | SemTS-10ep+A2 | SemTS-10ep+A3b |
                      SemTS-100ep岭 | SemTS-100ep+A2 | SemTS-100ep+A3b]，MSE。
用法: python scripts/aggregate_semts_improvements.py
"""
import json
import os
import re

LOG = os.path.join("experiments", "semts_improvement.log")
OUT = os.path.join("docs", "semts_improvement_results.md")
PAPER = os.path.join("docs", "shufflemtm_paper_numbers.json")
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
HORIZONS = ["96", "192", "336", "720"]


def parse_log():
    """{tag: {ds: {protocol: {H: mse}}}}；tag = b1_<ds> / tables_<ds>。"""
    res, cur_tag, cur_ds, cur_proto = {}, None, None, None
    if not os.path.exists(LOG):
        return res
    for line in open(LOG, encoding="utf-8", errors="replace"):
        m = re.search(r"===== (ridge|ft|softcc) (b1_\w+|tables_\w+) =====", line)
        if m:
            cur_proto, cur_tag = m.group(1), m.group(2)
            cur_ds = cur_tag.split("_", 1)[1]
            res.setdefault(cur_tag, {}).setdefault(cur_ds, {}).setdefault(cur_proto, {})
            continue
        m = re.search(r"H=\s*(\d+): MSE=([\d.]+)", line)
        if m and cur_tag:
            res[cur_tag][cur_ds][cur_proto][m.group(1)] = float(m.group(2))
            continue
        m = re.search(r"H=\s*(\d+): FT-MSE=([\d.]+)", line)
        if m and cur_tag:
            res[cur_tag][cur_ds][cur_proto][m.group(1)] = float(m.group(2))
            continue
        m = re.search(r"H=\s*(\d+): soft-channel ridge MSE=([\d.]+)", line)
        if m and cur_tag:
            res[cur_tag][cur_ds][cur_proto][m.group(1)] = float(m.group(2))
    return res


def main():
    res = parse_log()
    paper = json.load(open(PAPER, encoding="utf-8"))["table1_in_domain_forecast"]
    L = []
    a = L.append
    a("# SemTS 改进运动结果汇总")
    a("")
    a("> 生成：`python scripts/aggregate_semts_improvements.py`（数据源 experiments/semts_improvement.log）。")
    a("> 协议：岭 = 冻结+岭回归（原统一协议）；A2 = 冻结LP10+全量FT20（与 ShuffleMTM 同口径）；")
    a("> A3b = 软通道选择岭（DUET CCM 式，评估期改进）。10ep = 原预训练；100ep = B1 重训。")
    a("")
    for ds in DATASETS:
        a(f"## {ds}")
        a("")
        head = ("| H | ShuffleMTM(论文) | 10ep-岭 | 10ep+A2 | 10ep+A3b | "
                "100ep-岭 | 100ep+A2 | 100ep+A3b |")
        a(head)
        a("|---|" + "---|" * 7)
        for H in HORIZONS:
            cells = [fmt(paper[ds][H]["ShuffleMTM"][0])]
            for tag in ("tables", "b1"):
                for proto in ("ridge", "ft", "softcc"):
                    v = res.get(f"{tag}_{ds}", {}).get(ds, {}).get(proto, {}).get(H)
                    cells.append(f"**{v:.4f}**" if v is not None else "-")
            a("| " + H + " | " + " | ".join(cells) + " |")
        a("")
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"[aggregate] -> {OUT}")


def fmt(v):
    return f"{v:.3f}"


if __name__ == "__main__":
    main()
