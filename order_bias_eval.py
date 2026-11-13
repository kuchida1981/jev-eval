"""選択肢の順序(criteria dict の並び)を入れ替えたとき、decision モデルの回答が変わるかを測定する。

対象:
- choice 型の質問: criteria dict の並び順を forward/reversed で入れ替え
- noul 型の質問: 「yes/no の2択 choice」に変換した上で forward=(yes,no) / reversed=(no,yes) で入れ替え
  (doc の要望「yes|no を no|yes へ入れ替える」に対応)
- score 型は尺度の意味そのものが順序なので対象外(メモに明記)

1 (item, qid) あたり forward/reversed を単独の質問として送信し、他の質問からの干渉を避ける。
"""
import argparse
import json
import os
import time

from runner import ask

HERE = os.path.dirname(__file__)


def reversed_choice(q):
    items = list(q["criteria"].items())
    return {"type": "choice", "instructions": q["instructions"], "criteria": dict(reversed(items))}


def yesno_choice(q, order):
    desc = {"yes": "Yes / true.", "no": "No / false."}
    criteria = {k: desc[k] for k in order}
    return {"type": "choice", "instructions": q["instructions"], "criteria": criteria}


def build_variants(qid, q, gold):
    if q["type"] == "choice":
        return {
            "forward": q,
            "reversed": reversed_choice(q),
        }, gold
    if q["type"] == "noul":
        gold_label = "yes" if gold else "no"
        return {
            "forward": yesno_choice(q, ("yes", "no")),
            "reversed": yesno_choice(q, ("no", "yes")),
        }, gold_label
    return None, None  # score: 対象外


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://ubuntu:11434")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval_set.jsonl"))
    ap.add_argument("--models", nargs="*", default=["nimble", "tev1", "tev1:0.8b"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "order_bias_raw.jsonl"))
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.data, encoding="utf-8")]

    jobs = []
    for item in items:
        for qid, q in item["questions"].items():
            variants, gold_label = build_variants(qid, q, item["gold"].get(qid))
            if variants is None:
                continue
            jobs.append((item, qid, q["type"], variants, gold_label))

    total = len(jobs) * len(args.models) * 2
    done = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for item, qid, orig_type, variants, gold_label in jobs:
            for model in args.models:
                for order_name, variant_q in variants.items():
                    t0 = time.monotonic()
                    result = ask(args.base_url, model, item["state"], {qid: variant_q})
                    latency = time.monotonic() - t0
                    choice = None
                    if result["ok"]:
                        ans = result["raw"].get("answers", {}).get(qid, {})
                        choice = ans.get("choice")
                    record = {
                        "item_id": item["id"], "qid": qid, "orig_type": orig_type,
                        "model": model, "order": order_name, "ok": result["ok"],
                        "choice": choice, "gold_label": gold_label,
                        "latency_s": latency, "error": result["error"],
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done += 1
                    if done % 40 == 0 or done == total:
                        print(f"[{done}/{total}]")

    print(f"wrote {done} records -> {args.out}")


if __name__ == "__main__":
    main()
