"""order_bias_eval.py の出力を集計し、フリップ率・McNemar検定結果を表示する。"""
import argparse
import json
import os
from collections import defaultdict

from scipy.stats import chi2

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "order_bias_raw.jsonl"))
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.raw, encoding="utf-8")]
    by_key = defaultdict(dict)  # (item_id, qid, model) -> {order: record}
    for r in records:
        by_key[(r["item_id"], r["qid"], r["model"])][r["order"]] = r

    rows = []
    for (item_id, qid, model), d in by_key.items():
        fwd, rev = d.get("forward"), d.get("reversed")
        if not fwd or not rev or not fwd["ok"] or not rev["ok"]:
            continue
        rows.append({
            "item_id": item_id, "qid": qid, "model": model, "orig_type": fwd["orig_type"],
            "fwd_choice": fwd["choice"], "rev_choice": rev["choice"], "gold": fwd["gold_label"],
            "flipped": fwd["choice"] != rev["choice"],
            "fwd_correct": fwd["choice"] == fwd["gold_label"],
            "rev_correct": rev["choice"] == rev["gold_label"],
        })

    def summarize(rows, label):
        if not rows:
            print(f"{label}: no data")
            return
        n = len(rows)
        n_flip = sum(r["flipped"] for r in rows)
        acc_fwd = sum(r["fwd_correct"] for r in rows) / n
        acc_rev = sum(r["rev_correct"] for r in rows) / n
        b = sum(r["fwd_correct"] and not r["rev_correct"] for r in rows)  # fwd correct, rev wrong
        c = sum((not r["fwd_correct"]) and r["rev_correct"] for r in rows)  # fwd wrong, rev correct
        if b + c > 0:
            stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected McNemar
            p = chi2.sf(stat, df=1)
        else:
            stat, p = 0.0, 1.0
        print(f"\n== {label} (n={n}) ==")
        print(f"  flip rate: {n_flip}/{n} = {n_flip/n:.3f}")
        print(f"  accuracy forward={acc_fwd:.3f}  reversed={acc_rev:.3f}")
        print(f"  McNemar: b(fwd-only correct)={b} c(rev-only correct)={c} chi2={stat:.3f} p={p:.4f}"
              f"  {'=> SIGNIFICANT (p<0.05)' if p < 0.05 else '=> not significant'}")
        return {"n": n, "flip_rate": n_flip / n, "acc_fwd": acc_fwd, "acc_rev": acc_rev,
                "b": b, "c": c, "chi2": stat, "p": p}

    print("=" * 60)
    overall = summarize(rows, "全体")
    by_type = {}
    for t in ("choice", "noul"):
        by_type[t] = summarize([r for r in rows if r["orig_type"] == t], f"元の質問型 = {t}")
    by_model = {}
    for m in sorted({r["model"] for r in rows}):
        by_model[m] = summarize([r for r in rows if r["model"] == m], f"model = {m}")

    print("\nフリップした具体例(最大10件):")
    flipped = [r for r in rows if r["flipped"]][:10]
    for r in flipped:
        print(f"  {r['item_id']} / {r['qid']} / {r['model']}: "
              f"forward={r['fwd_choice']} -> reversed={r['rev_choice']} (gold={r['gold']})")

    out = {"overall": overall, "by_type": by_type, "by_model": by_model,
           "flipped_examples": flipped}
    out_path = os.path.join(HERE, "results", "order_bias_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
