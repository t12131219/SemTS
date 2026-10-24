"""把 SemTS 分类探针结果注入 shufflemtm_tables_results.md 的 Table 3/4。

用法: python scripts/patch_semts_classify_rows.py [--md docs/shufflemtm_tables_results.md]
     [--json_dir experiments/semts_classify_probe]
"""
import argparse
import json
import os
import re


def pct(v):
    return f"{v * 100:.3f}" if v is not None and v <= 1 else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=str, default="docs/shufflemtm_tables_results.md")
    ap.add_argument("--json_dir", type=str, default="experiments/semts_classify_probe")
    args = ap.parse_args()

    semts = {}
    for name in ("AD", "PTB"):
        p = os.path.join(args.json_dir, f"{name}.json")
        if os.path.exists(p):
            semts[name] = json.load(open(p, encoding="utf-8"))

    text = open(args.md, encoding="utf-8").read()

    # ===== Table 3：每个数据集小节末尾追加 SemTS 行 =====
    for name in ("AD", "PTB"):
        if name not in semts:
            continue
        m = semts[name]["1.0"]
        row = (f"| **SemTS** | **{pct(m['acc'])}** | **{pct(m['prec'])}** | "
               f"**{pct(m['rec'])}** | **{pct(m['f1'])}** |\n")
        # 在 "### {name}" 小节中 COMET 行之后插入（COMET 是该小节最后一个基线行）
        idx = text.find(f"### {name}\n")
        assert idx >= 0, name
        sub_start = idx
        sub_end = text.find("### ", sub_start + 4)
        if sub_end < 0:
            sub_end = text.find("## Table 4", sub_start)
        sub = text[sub_start:sub_end]
        comet = sub.rfind("| COMET |")
        insert_at = sub_start + comet + len(sub[comet:].split("\n")[0]) + 1
        text = text[:insert_at] + row + text[insert_at:]

    # ===== Table 4：每个比例小节末尾追加 SemTS 行 =====
    for frac_key, frac_label in (("0.2", "20%"), ("0.1", "10%"), ("0.05", "5%")):
        sec = f"### {frac_label}\n"
        idx = text.find(sec)
        if idx < 0:
            continue
        sub_start = idx
        sub_end = text.find("### ", sub_start + 4)
        if sub_end < 0:
            sub_end = len(text)
        sub = text[sub_start:sub_end]
        comet = sub.rfind("| COMET |")
        insert_at = sub_start + comet + len(sub[comet:].split("\n")[0]) + 1
        ad = semts.get("AD", {}).get(frac_key, {}).get("acc")
        ptb = semts.get("PTB", {}).get(frac_key, {}).get("acc")
        cad = pct(ad) if ad is not None else "-"
        cpt = pct(ptb) if ptb is not None else "-"
        row = f"| **SemTS** | **{cad}** | **{cpt}** |\n"
        text = text[:insert_at] + row + text[insert_at:]

    open(args.md, "w", encoding="utf-8").write(text)
    print(f"[patch] SemTS rows injected -> {args.md}")


if __name__ == "__main__":
    main()
