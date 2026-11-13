"""Step3-a 集計: 同一入力・5回繰り返し実行の一致率と確率のばらつき。"""
import json
import os
import statistics as stats
from collections import Counter, defaultdict

HERE = os.path.dirname(__file__)


def main():
    items_by_id = {}
    for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        items_by_id[d["id"]] = d

    recs = [json.loads(l) for l in open(os.path.join(HERE, "results", "phase2", "step3_stability_raw.jsonl"),
                                          encoding="utf-8")]

    # (item_id, qid, model) -> list of (answer, confidence) across 5 reps
    by_key = defaultdict(list)
    for r in recs:
        if not r["ok"]:
            continue
        item = items_by_id[r["item_id"]]
        for qid in item["questions"]:
            by_key[(r["item_id"], qid, r["model"])].append(
                (r["answers"].get(qid), (r["confidences"] or {}).get(qid)))

    rows = []
    for (item_id, qid, model), vals in by_key.items():
        answers = [v[0] for v in vals]
        confs = [v[1] for v in vals if v[1] is not None]
        n = len(answers)
        counts = Counter(answers)
        top_answer, top_count = counts.most_common(1)[0]
        agreement = top_count / n
        conf_std = stats.pstdev(confs) if len(confs) >= 2 else 0.0
        item = items_by_id[item_id]
        rows.append({
            "item_id": item_id, "qid": qid, "model": model, "task": item["task"],
            "qtype": item["questions"][qid]["type"], "n": n, "agreement": agreement,
            "unanimous": top_count == n, "conf_std": conf_std, "answers": answers,
        })

    def summarize(rows, label):
        n = len(rows)
        mean_agree = sum(r["agreement"] for r in rows) / n
        pct_unanimous = sum(r["unanimous"] for r in rows) / n
        mean_conf_std = sum(r["conf_std"] for r in rows) / n
        return {"n": n, "mean_agreement": mean_agree, "pct_unanimous": pct_unanimous,
                "mean_conf_std": mean_conf_std}

    overall = summarize(rows, "全体")
    by_model = {m: summarize([r for r in rows if r["model"] == m], m)
                for m in sorted({r["model"] for r in rows})}
    by_qtype = {t: summarize([r for r in rows if r["qtype"] == t], t)
                for t in sorted({r["qtype"] for r in rows})}
    by_task = {t: summarize([r for r in rows if r["task"] == t], t)
               for t in sorted({r["task"] for r in rows})}

    unstable = sorted([r for r in rows if not r["unanimous"]], key=lambda r: r["agreement"])

    lines = ["# Step3-a: 安定性測定(同一入力5回繰り返し実行)\n"]
    lines.append(f"対象: 全48シナリオ × 3モデル × 5回 = {overall['n']} 組(item, qid, model)\n")

    lines.append("## 全体\n")
    lines.append(f"- 平均一致率(5回中最頻回答の割合): {overall['mean_agreement']:.3f}")
    lines.append(f"- 完全一致(5/5)の割合: {overall['pct_unanimous']:.3f}")
    lines.append(f"- 確信度の標準偏差(平均): {overall['mean_conf_std']:.3f}\n")

    lines.append("## モデル別\n")
    lines.append("| model | n | 平均一致率 | 完全一致率 | 確信度std(平均) |")
    lines.append("|---|---|---|---|---|")
    for m, s in by_model.items():
        lines.append(f"| {m} | {s['n']} | {s['mean_agreement']:.3f} | {s['pct_unanimous']:.3f} | "
                      f"{s['mean_conf_std']:.3f} |")
    lines.append("")

    lines.append("## 質問型別\n")
    lines.append("| qtype | n | 平均一致率 | 完全一致率 | 確信度std(平均) |")
    lines.append("|---|---|---|---|---|")
    for t, s in by_qtype.items():
        lines.append(f"| {t} | {s['n']} | {s['mean_agreement']:.3f} | {s['pct_unanimous']:.3f} | "
                      f"{s['mean_conf_std']:.3f} |")
    lines.append("")

    lines.append("## タスク別\n")
    lines.append("| task | n | 平均一致率 | 完全一致率 | 確信度std(平均) |")
    lines.append("|---|---|---|---|---|")
    for t, s in by_task.items():
        lines.append(f"| {t} | {s['n']} | {s['mean_agreement']:.3f} | {s['pct_unanimous']:.3f} | "
                      f"{s['mean_conf_std']:.3f} |")
    lines.append("")

    lines.append(f"## 不安定な組み合わせ(完全一致でないもの、{len(unstable)}/{overall['n']}件中、agreement低い順に最大15件)\n")
    lines.append("| item_id | qid | model | agreement | 回答の内訳 |")
    lines.append("|---|---|---|---|---|")
    for r in unstable[:15]:
        counts = Counter(r["answers"])
        breakdown = ", ".join(f"{k}:{v}" for k, v in counts.most_common())
        lines.append(f"| {r['item_id']} | {r['qid']} | {r['model']} | {r['agreement']:.2f} | {breakdown} |")
    lines.append("")

    lines.append("## 所見\n")
    lines.append("- (下書き。数値確認の上で執筆)\n")

    out_path = os.path.join(HERE, "results", "phase2", "step3a_stability.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")
    print(json.dumps({"overall": overall, "by_model": by_model, "by_qtype": by_qtype, "by_task": by_task},
                      indent=2))


if __name__ == "__main__":
    main()
