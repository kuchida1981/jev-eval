"""Step1: qwen3.5:9b がタイムアウトした7シナリオを全モデルから除外し、
   「全モデルが回答できた共通問題セット」だけで正解率を再集計し、元の集計との差分を出す。
"""
import json
import os
from collections import defaultdict

from analyze import load_items_by_id, flatten, accuracy, group_accuracy

HERE = os.path.dirname(__file__)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    items_by_id = load_items_by_id(os.path.join(HERE, "data", "eval_set.jsonl"))
    records = load_records(os.path.join(HERE, "results", "raw_full.jsonl"))

    failed_item_ids = {r["item_id"] for r in records if not r["ok"]}
    print(f"除外対象シナリオ(いずれかのモデルでエラー): {len(failed_item_ids)}件")
    for i in sorted(failed_item_ids):
        print(" -", i)

    rows_all = flatten(records, items_by_id)
    rows_common = [r for r in rows_all if r["item_id"] not in failed_item_ids]

    models = sorted({r["model"] for r in rows_all})

    def report(rows, label):
        out = {}
        acc_all, nc, n = accuracy(rows)
        out["_overall"] = {}
        for m in models:
            mrows = [r for r in rows if r["model"] == m]
            a, c, t = accuracy(mrows)
            out["_overall"][m] = (a, c, t)
        for key in ("difficulty", "qtype", "task", "lang"):
            out[key] = {"_keys": sorted({r[key] for r in rows if r[key] is not None})}
            for m in models:
                out[key][m] = group_accuracy([r for r in rows if r["model"] == m], key)
        return out

    full = report(rows_all, "全件(元の集計)")
    common = report(rows_common, "共通問題セット(timeout7件除外)")

    lines = ["# Step1: 共通問題セットでの公平比較\n"]
    lines.append(f"timeout等で除外したシナリオ: {len(failed_item_ids)}件 "
                  f"({', '.join(sorted(failed_item_ids))})\n")
    n_common_items = len({r['item_id'] for r in rows_common})
    lines.append(f"共通問題セットのシナリオ数: {len(items_by_id) - len(failed_item_ids)} / {len(items_by_id)}\n")

    lines.append("## 総合正解率: 全件 vs 共通セット(差分)\n")
    lines.append("| model | 全件(元) | 共通セット | 差分 |")
    lines.append("|---|---|---|---|")
    for m in models:
        a_full, c_full, t_full = full["_overall"][m]
        a_common, c_common, t_common = common["_overall"][m]
        diff = a_common - a_full
        lines.append(f"| {m} | {a_full:.3f} ({c_full}/{t_full}) | {a_common:.3f} ({c_common}/{t_common}) | "
                      f"{diff:+.3f} |")
    lines.append("")

    for key, label in (("difficulty", "難易度別"), ("qtype", "質問型別"), ("task", "タスク別"), ("lang", "言語別")):
        keys = common[key]["_keys"]
        lines.append(f"## {label}: 全件 vs 共通セット\n")
        header = "| model | " + " | ".join(f"{k}(全/共通)" for k in keys) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(keys))
        for m in models:
            cells = []
            for k in keys:
                a_full = full[key][m].get(k, (float("nan"), 0))[0]
                a_common = common[key][m].get(k, (float("nan"), 0))[0]
                cells.append(f"{a_full:.3f}/{a_common:.3f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 所見\n")
    lines.append("- (自動生成の下書き。数値を確認して埋めること)\n")

    out_path = os.path.join(HERE, "results", "phase2", "step1_common_subset.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
