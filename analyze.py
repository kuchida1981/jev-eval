"""run_eval.py の出力 (results/raw_*.jsonl) を集計し、summary.md と図を生成する。

使い方:
  .venv/bin/python analyze.py results/raw_full.jsonl --out-dir results
"""
import argparse
import json
import os
import statistics as stats
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)


def load_items_by_id(path):
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f}


def load_records(paths):
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def flatten(records, items_by_id):
    """1 qid = 1 行の flat レコードに展開する。"""
    rows = []
    for r in records:
        item = items_by_id.get(r["item_id"])
        qtypes = {qid: q["type"] for qid, q in item["questions"].items()} if item else {}
        if not r["ok"]:
            rows.append(dict(model=r["model"], mode=r["mode"], item_id=r["item_id"], task=r["task"],
                              lang=r["lang"], difficulty=r["difficulty"], qid=None, qtype=None,
                              value=None, gold=None, confidence=None, correct=None,
                              latency_s=r["latency_s"], error=r["error"]))
            continue
        for qid, correct in (r["correct"] or {}).items():
            rows.append(dict(
                model=r["model"], mode=r["mode"], item_id=r["item_id"], task=r["task"],
                lang=r["lang"], difficulty=r["difficulty"], qid=qid, qtype=qtypes.get(qid),
                value=(r["answers"] or {}).get(qid), gold=r["gold"].get(qid),
                confidence=(r["confidences"] or {}).get(qid), correct=correct,
                latency_s=r["latency_s"], error=None,
            ))
    return rows


def accuracy(rows):
    graded = [r for r in rows if r["correct"] is not None]
    if not graded:
        return float("nan"), 0, 0
    n_correct = sum(r["correct"] for r in graded)
    return n_correct / len(graded), n_correct, len(graded)


def group_accuracy(rows, key):
    buckets = defaultdict(list)
    for r in rows:
        if r["correct"] is not None:
            buckets[r[key]].append(r["correct"])
    return {k: (sum(v) / len(v), len(v)) for k, v in buckets.items()}


def fleiss_kappa(item_answers):
    """item_answers: list of {model: value} (categorical) for the same (item_id, qid).
    モデル数が一定でない場合も対応(欠損は無視)。"""
    all_models = sorted({m for ans in item_answers for m in ans})
    # 質問型が混在(choice=str / noul=bool / score=str)するため str() で統一して比較する
    categories = sorted({str(v) for ans in item_answers for v in ans.values() if v is not None})
    if len(categories) < 2 or not item_answers:
        return float("nan"), 0
    n_items = 0
    p_j = defaultdict(float)
    row_agreements = []
    total_raters = 0
    for ans in item_answers:
        vals = [str(v) for v in ans.values() if v is not None]
        n = len(vals)
        if n < 2:
            continue
        n_items += 1
        total_raters += n
        counts = defaultdict(int)
        for v in vals:
            counts[v] += 1
        for c in categories:
            p_j[c] += counts.get(c, 0)
        agree = sum(c * (c - 1) for c in counts.values())
        row_agreements.append(agree / (n * (n - 1)))
    if n_items == 0:
        return float("nan"), 0
    p_bar = sum(row_agreements) / n_items
    p_j_norm = {c: v / total_raters for c, v in p_j.items()}
    pe_bar = sum(p ** 2 for p in p_j_norm.values())
    if pe_bar >= 1:
        return float("nan"), n_items
    kappa = (p_bar - pe_bar) / (1 - pe_bar)
    return kappa, n_items


def model_agreement(rows, models):
    by_key = defaultdict(dict)
    for r in rows:
        if r["model"] in models and r["qid"] is not None:
            by_key[(r["item_id"], r["qid"])][r["model"]] = r["value"]
    item_answers = list(by_key.values())
    kappa, n = fleiss_kappa(item_answers)

    all_agree_correct = 0
    all_agree_wrong = 0
    total_full = 0
    gold_by_key = {}
    for r in rows:
        if r["qid"] is not None:
            gold_by_key[(r["item_id"], r["qid"])] = r["gold"]
    for key, ans in by_key.items():
        vals = [ans.get(m) for m in models]
        if any(v is None for v in vals):
            continue
        total_full += 1
        if len(set(vals)) == 1:
            if vals[0] == gold_by_key.get(key):
                all_agree_correct += 1
            else:
                all_agree_wrong += 1
    return {
        "fleiss_kappa": kappa, "n_items_for_kappa": n,
        "n_fully_answered": total_full,
        "all_agree_and_correct": all_agree_correct,
        "all_agree_and_wrong": all_agree_wrong,
    }


def ece(rows, n_bins=5):
    graded = [r for r in rows if r["correct"] is not None and r["confidence"] is not None]
    if not graded:
        return float("nan"), []
    graded.sort(key=lambda r: r["confidence"])
    bin_size = max(1, len(graded) // n_bins)
    bins = []
    total_err = 0.0
    for i in range(0, len(graded), bin_size):
        chunk = graded[i:i + bin_size]
        if not chunk:
            continue
        avg_conf = sum(r["confidence"] for r in chunk) / len(chunk)
        avg_acc = sum(r["correct"] for r in chunk) / len(chunk)
        bins.append({"n": len(chunk), "avg_confidence": avg_conf, "accuracy": avg_acc})
        total_err += abs(avg_conf - avg_acc) * len(chunk)
    return total_err / len(graded), bins


def coverage_accuracy_curve(rows, thresholds=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    graded = [r for r in rows if r["correct"] is not None and r["confidence"] is not None]
    out = []
    for t in thresholds:
        kept = [r for r in graded if r["confidence"] >= t]
        if not kept:
            out.append({"threshold": t, "coverage": 0.0, "accuracy": float("nan")})
            continue
        out.append({
            "threshold": t,
            "coverage": len(kept) / len(graded),
            "accuracy": sum(r["correct"] for r in kept) / len(kept),
        })
    return out


def latency_stats(rows):
    """rows は qid 単位に展開済みなので、request(item) 単位に一度だけ数える。"""
    seen = {}
    for r in rows:
        seen[(r["model"], r["item_id"])] = r["latency_s"]
    by_model = defaultdict(list)
    for (model, _item_id), lat in seen.items():
        by_model[model].append(lat)
    out = {}
    for m, lats in by_model.items():
        lats_sorted = sorted(lats)
        p95_idx = max(0, int(len(lats_sorted) * 0.95) - 1)
        out[m] = {
            "median_s": stats.median(lats_sorted),
            "p95_s": lats_sorted[p95_idx],
            "max_s": max(lats_sorted),
            "min_s": min(lats_sorted),
            "n": len(lats_sorted),
        }
    return out


def write_summary_md(path, models, overall_acc, by_difficulty, by_task, by_qtype, by_lang,
                      agreement, ece_by_model, cov_curve_by_model, lat_stats, common_wrong,
                      reference_note, error_counts):
    lines = ["# Jev 代替決定モデル ローカル評価 サマリー\n"]

    lines.append("## モデル別 総合正解率・エラー率\n")
    lines.append("| model | accuracy | n(graded) | errors | error rate |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        acc, nc, n = overall_acc[m]
        n_err, n_total = error_counts.get(m, (0, 0))
        err_rate = n_err / n_total if n_total else float("nan")
        lines.append(f"| {m} | {acc:.3f} | {nc}/{n} | {n_err}/{n_total} | {err_rate:.3f} |")
    lines.append("")

    lines.append("## A. 難易度別正解率(easy/medium/hard)\n")
    lines.append("| model | " + " | ".join(by_difficulty["_keys"]) + " |")
    lines.append("|---|" + "---|" * len(by_difficulty["_keys"]))
    for m in models:
        row = [f"{by_difficulty[m].get(k, (float('nan'), 0))[0]:.3f}" for k in by_difficulty["_keys"]]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## B-1. 質問型別正解率 (choice/noul/score)\n")
    lines.append("| model | " + " | ".join(by_qtype["_keys"]) + " |")
    lines.append("|---|" + "---|" * len(by_qtype["_keys"]))
    for m in models:
        row = [f"{by_qtype[m].get(k, (float('nan'), 0))[0]:.3f}" for k in by_qtype["_keys"]]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## B-2. 言語別正解率 (ja/en)\n")
    lines.append("| model | " + " | ".join(by_lang["_keys"]) + " |")
    lines.append("|---|" + "---|" * len(by_lang["_keys"]))
    for m in models:
        row = [f"{by_lang[m].get(k, (float('nan'), 0))[0]:.3f}" for k in by_lang["_keys"]]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## B-3. タスク別正解率\n")
    lines.append("| model | " + " | ".join(by_task["_keys"]) + " |")
    lines.append("|---|" + "---|" * len(by_task["_keys"]))
    for m in models:
        row = [f"{by_task[m].get(k, (float('nan'), 0))[0]:.3f}" for k in by_task["_keys"]]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## B-4. decision モデル間の一致度\n")
    lines.append(f"- Fleiss' kappa: {agreement['fleiss_kappa']:.3f} (n={agreement['n_items_for_kappa']})")
    lines.append(f"- 全モデル一致・全数回答できた質問: {agreement['n_fully_answered']}")
    lines.append(f"- 全モデル一致 かつ 正解: {agreement['all_agree_and_correct']}")
    lines.append(f"- 全モデル一致 かつ 誤答(ラベル側の問題の可能性あり): {agreement['all_agree_and_wrong']}")
    lines.append("")

    if common_wrong:
        lines.append("### 全モデルが同じ誤答をした項目(要確認)\n")
        for c in common_wrong:
            lines.append(f"- `{c['item_id']}` / `{c['qid']}`: 全モデルが `{c['value']}` と回答 "
                          f"(gold=`{c['gold']}`)")
        lines.append("")

    lines.append("## C. キャリブレーション(decision モデルのみ、baseline は確率非対応)\n")
    for m, e in ece_by_model.items():
        lines.append(f"### {m} — ECE = {e:.3f}\n")
        lines.append("| bin avg confidence | bin accuracy | n |")
        lines.append("|---|---|---|")
        for b in cov_curve_by_model.get(m, {}).get("bins", []):
            lines.append(f"| {b['avg_confidence']:.3f} | {b['accuracy']:.3f} | {b['n']} |")
        lines.append("")
        lines.append("閾値 vs 正解率/回答できる割合:")
        lines.append("| threshold | coverage | accuracy |")
        lines.append("|---|---|---|")
        for c in cov_curve_by_model.get(m, {}).get("curve", []):
            lines.append(f"| {c['threshold']} | {c['coverage']:.3f} | {c['accuracy']:.3f} |")
        lines.append("")

    lines.append("## D. レイテンシ(秒)\n")
    lines.append("| model | median | p95 | min | max | n |")
    lines.append("|---|---|---|---|---|---|")
    for m, s in lat_stats.items():
        lines.append(f"| {m} | {s['median_s']:.2f} | {s['p95_s']:.2f} | {s['min_s']:.2f} | "
                      f"{s['max_s']:.2f} | {s['n']} |")
    lines.append("")

    lines.append("## 参考: 他者申告の精度(未検証、ベンダー自己申告またはサードパーティ記事)\n")
    lines.append(reference_note)
    lines.append("")

    lines.append("## 今回のスコープで未実施(次フェーズ)\n")
    lines.append("- 選択肢シャッフルによる順序バイアス測定\n"
                  "- 質問の言い換え・繰り返し実行による安定性測定\n"
                  "- 選択肢数(3/10/50)バリエーション\n"
                  "- llama.cpp 経由モデル(d3-mini 等)、ホスト版 Jev API との比較\n"
                  "- VRAM 実測(`/api/ps` ベース)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval_set.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    items_by_id = load_items_by_id(args.data)
    records = load_records(args.results)
    rows = flatten(records, items_by_id)
    models = sorted({r["model"] for r in rows})
    decision_models = sorted({r["model"] for r in records if r["mode"] == "decision"})

    overall_acc = {m: accuracy([r for r in rows if r["model"] == m]) for m in models}

    def grouped(key):
        out = {"_keys": sorted({r[key] for r in rows if r[key] is not None})}
        for m in models:
            out[m] = group_accuracy([r for r in rows if r["model"] == m], key)
        return out

    by_difficulty = grouped("difficulty")
    by_task = grouped("task")
    by_qtype = grouped("qtype")
    by_lang = grouped("lang")

    agreement = model_agreement(rows, decision_models)

    # 全モデル共通誤答の抽出
    by_key = defaultdict(dict)
    gold_by_key = {}
    for r in rows:
        if r["qid"] is not None:
            by_key[(r["item_id"], r["qid"])][r["model"]] = r["value"]
            gold_by_key[(r["item_id"], r["qid"])] = r["gold"]
    common_wrong = []
    for (item_id, qid), ans in by_key.items():
        vals = [ans.get(m) for m in decision_models]
        if all(v is not None for v in vals) and len(set(vals)) == 1 and vals[0] != gold_by_key[(item_id, qid)]:
            common_wrong.append({"item_id": item_id, "qid": qid, "value": vals[0], "gold": gold_by_key[(item_id, qid)]})

    ece_by_model = {}
    cov_curve_by_model = {}
    for m in decision_models:
        mrows = [r for r in rows if r["model"] == m]
        e, bins = ece(mrows)
        ece_by_model[m] = e
        cov_curve_by_model[m] = {"bins": bins, "curve": coverage_accuracy_curve(mrows)}

    lat_stats = latency_stats(rows)

    error_counts = {}
    for m in models:
        m_records = [r for r in records if r["model"] == m]
        n_err = sum(1 for r in m_records if not r["ok"])
        error_counts[m] = (n_err, len(m_records))

    os.makedirs(args.out_dir, exist_ok=True)

    reference_note = (
        "- Nimble 75.7% / Jev 1.13 76.0% / Tev1 4B 73.3% / Tev1 0.8B 63.5% "
        "(Ollama ブログ記事記載の 13 データセット・3,880 判断の平均精度。自前計測ではない)\n"
        "- 今回の自前データセット(48シナリオ)は人間でも割れる曖昧ケースを含む少数データのため、"
        "上記の大規模ベンチマークと直接比較しないこと。"
    )

    write_summary_md(
        os.path.join(args.out_dir, "summary.md"), models, overall_acc, by_difficulty,
        by_task, by_qtype, by_lang, agreement, ece_by_model, cov_curve_by_model,
        lat_stats, common_wrong, reference_note, error_counts,
    )

    # 精度 x 速度 散布図
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in models:
        acc, _, _ = overall_acc[m]
        med = lat_stats[m]["median_s"]
        ax.scatter(med, acc, s=80)
        ax.annotate(m, (med, acc), textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_xlabel("median latency (s, log scale)")
    ax.set_ylabel("overall accuracy")
    ax.set_title("Accuracy vs Latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "accuracy_vs_latency.png"), dpi=150)

    # キャリブレーション曲線
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for m in decision_models:
        bins = cov_curve_by_model[m]["bins"]
        if not bins:
            continue
        xs = [b["avg_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("avg confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title("Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "calibration.png"), dpi=150)

    print("wrote:")
    print(" -", os.path.join(args.out_dir, "summary.md"))
    print(" -", os.path.join(args.out_dir, "accuracy_vs_latency.png"))
    print(" -", os.path.join(args.out_dir, "calibration.png"))
    for m in models:
        acc, nc, n = overall_acc[m]
        print(f"   {m}: acc={acc:.3f} ({nc}/{n}) median_latency={lat_stats[m]['median_s']:.2f}s")


if __name__ == "__main__":
    main()
