"""Step3-b 集計: 質問文の言い換え2パターンで回答・正解率が変わる割合。"""
import json
import os
from collections import defaultdict

from scipy.stats import chi2

HERE = os.path.dirname(__file__)


def mcnemar(rows, key_a, key_b):
    b = sum(r[key_a] and not r[key_b] for r in rows)
    c = sum((not r[key_a]) and r[key_b] for r in rows)
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return stat, chi2.sf(stat, df=1), b, c


def main():
    items_by_id = {}
    for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        items_by_id[d["id"]] = d

    recs = [json.loads(l) for l in open(os.path.join(HERE, "results", "phase2", "step3_paraphrase_raw.jsonl"),
                                          encoding="utf-8")]
    by_key = defaultdict(dict)
    for r in recs:
        by_key[(r["item_id"], r["qid"], r["model"])][r["version"]] = r

    rows = []
    for (item_id, qid, model), d in by_key.items():
        orig, p1, p2 = d.get("orig"), d.get("p1"), d.get("p2")
        if not (orig and p1 and p2) or not (orig["ok"] and p1["ok"] and p2["ok"]):
            continue
        item = items_by_id[item_id]
        rows.append({
            "item_id": item_id, "qid": qid, "model": model, "task": item["task"],
            "qtype": item["questions"][qid]["type"],
            "orig": orig["choice"], "p1": p1["choice"], "p2": p2["choice"], "gold": orig["gold"],
            "flip_p1": orig["choice"] != p1["choice"], "flip_p2": orig["choice"] != p2["choice"],
            "orig_c": orig["choice"] == orig["gold"], "p1_c": p1["choice"] == p1["gold"],
            "p2_c": p2["choice"] == p2["gold"],
        })

    def summarize(rows):
        n = len(rows)
        if n == 0:
            return None
        any_flip = sum(r["flip_p1"] or r["flip_p2"] for r in rows)
        acc_orig = sum(r["orig_c"] for r in rows) / n
        acc_p1 = sum(r["p1_c"] for r in rows) / n
        acc_p2 = sum(r["p2_c"] for r in rows) / n
        stat1, p1v, b1, c1 = mcnemar(rows, "orig_c", "p1_c")
        stat2, p2v, b2, c2 = mcnemar(rows, "orig_c", "p2_c")
        return {"n": n, "flip_p1_rate": sum(r["flip_p1"] for r in rows) / n,
                "flip_p2_rate": sum(r["flip_p2"] for r in rows) / n,
                "any_flip_rate": any_flip / n, "acc_orig": acc_orig, "acc_p1": acc_p1, "acc_p2": acc_p2,
                "mcnemar_p1_p": p1v, "mcnemar_p2_p": p2v}

    overall = summarize(rows)
    by_model = {m: summarize([r for r in rows if r["model"] == m]) for m in sorted({r["model"] for r in rows})}
    by_qtype = {t: summarize([r for r in rows if r["qtype"] == t]) for t in sorted({r["qtype"] for r in rows})}
    by_task = {t: summarize([r for r in rows if r["task"] == t]) for t in sorted({r["task"] for r in rows})}

    flipped_any = sorted([r for r in rows if r["flip_p1"] or r["flip_p2"]],
                          key=lambda r: r["item_id"])

    def fmt(s):
        if s is None:
            return "n/a"
        return (f"n={s['n']} flip(p1)={s['flip_p1_rate']:.3f} flip(p2)={s['flip_p2_rate']:.3f} "
                f"any_flip={s['any_flip_rate']:.3f} acc(orig/p1/p2)={s['acc_orig']:.3f}/"
                f"{s['acc_p1']:.3f}/{s['acc_p2']:.3f} McNemar p(p1)={s['mcnemar_p1_p']:.3f} "
                f"p(p2)={s['mcnemar_p2_p']:.3f}")

    lines = ["# Step3-b: 言い換えバリエーション測定\n"]
    lines.append(f"有効サンプル: {overall['n']} 組(item×qid×model)。orig/p1/p2 を単独質問として比較。\n")
    lines.append("## 全体\n")
    lines.append(f"- {fmt(overall)}\n")

    lines.append("## モデル別\n")
    lines.append("| model | n | flip(p1) | flip(p2) | any_flip | acc orig/p1/p2 | McNemar p(p1/p2) |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, s in by_model.items():
        lines.append(f"| {m} | {s['n']} | {s['flip_p1_rate']:.3f} | {s['flip_p2_rate']:.3f} | "
                      f"{s['any_flip_rate']:.3f} | {s['acc_orig']:.3f}/{s['acc_p1']:.3f}/{s['acc_p2']:.3f} | "
                      f"{s['mcnemar_p1_p']:.3f}/{s['mcnemar_p2_p']:.3f} |")
    lines.append("")

    lines.append("## 質問型別\n")
    lines.append("| qtype | n | flip(p1) | flip(p2) | any_flip | acc orig/p1/p2 |")
    lines.append("|---|---|---|---|---|---|")
    for t, s in by_qtype.items():
        lines.append(f"| {t} | {s['n']} | {s['flip_p1_rate']:.3f} | {s['flip_p2_rate']:.3f} | "
                      f"{s['any_flip_rate']:.3f} | {s['acc_orig']:.3f}/{s['acc_p1']:.3f}/{s['acc_p2']:.3f} |")
    lines.append("")

    lines.append("## タスク別\n")
    lines.append("| task | n | flip(p1) | flip(p2) | any_flip | acc orig/p1/p2 |")
    lines.append("|---|---|---|---|---|---|")
    for t, s in by_task.items():
        lines.append(f"| {t} | {s['n']} | {s['flip_p1_rate']:.3f} | {s['flip_p2_rate']:.3f} | "
                      f"{s['any_flip_rate']:.3f} | {s['acc_orig']:.3f}/{s['acc_p1']:.3f}/{s['acc_p2']:.3f} |")
    lines.append("")

    lines.append(f"## 言い換えで回答が変わった組み合わせ(any_flip, {len(flipped_any)}/{overall['n']}件中、最大15件)\n")
    lines.append("| item_id | qid | model | orig | p1 | p2 | gold |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in flipped_any[:15]:
        lines.append(f"| {r['item_id']} | {r['qid']} | {r['model']} | {r['orig']} | {r['p1']} | {r['p2']} | "
                      f"{r['gold']} |")
    lines.append("")

    lines.append("## 所見\n")
    lines.append("- (下書き。数値確認の上で執筆)\n")

    out_path = os.path.join(HERE, "results", "phase2", "step3b_paraphrase.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")
    print("overall:", fmt(overall))
    for t, s in by_task.items():
        print(t, fmt(s))


if __name__ == "__main__":
    main()
