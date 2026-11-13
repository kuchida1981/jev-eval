"""全モデル x 全評価項目を実行し、results/raw_<timestamp>.jsonl に保存する。

使い方:
  .venv/bin/python run_eval.py --base-url http://ubuntu:11434 \
      --decision nimble tev1 tev1:0.8b --baseline qwen3.5:9b \
      [--limit 10] [--out results/raw_smoke.jsonl]
"""
import argparse
import json
import os
import time

from runner import ask, ask_baseline
from extract import extract_answer, grade

HERE = os.path.dirname(__file__)


def load_items(path, limit=None):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    return items[:limit] if limit else items


def run_one(base_url, model, mode, item):
    fn = ask if mode == "decision" else ask_baseline
    result = fn(base_url, model, item["state"], item["questions"])
    record = {
        "item_id": item["id"],
        "task": item["task"],
        "lang": item["lang"],
        "difficulty": item["difficulty"],
        "model": model,
        "mode": mode,
        "ok": result["ok"],
        "latency_s": result["latency_s"],
        "error": result["error"],
    }
    if result["ok"]:
        answers, confidences = extract_answer(mode, result["raw"], item["questions"])
        record["answers"] = answers
        record["confidences"] = confidences
        record["correct"] = grade(answers, item["gold"])
    else:
        record["answers"] = None
        record["confidences"] = None
        record["correct"] = None
    record["gold"] = item["gold"]
    record["raw"] = result["raw"]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://ubuntu:11434")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval_set.jsonl"))
    ap.add_argument("--decision", nargs="*", default=[], help="decision モデル名(/v1/systemone)")
    ap.add_argument("--baseline", nargs="*", default=[], help="baseline モデル名(/api/chat 経由)")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 件だけ実行(スモークテスト用)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = load_items(args.data, args.limit)
    out_path = args.out or os.path.join(HERE, "results", f"raw_{int(time.time())}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    runs = [(m, "decision") for m in args.decision] + [(m, "baseline") for m in args.baseline]
    total = len(items) * len(runs)
    done = 0
    n_correct_overall = 0
    n_graded_overall = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for model, mode in runs:
            n_correct, n_graded, n_err = 0, 0, 0
            for item in items:
                record = run_one(args.base_url, model, mode, item)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1
                if record["ok"]:
                    for v in (record["correct"] or {}).values():
                        n_graded += 1
                        n_correct += int(v)
                else:
                    n_err += 1
                if done % 10 == 0 or done == total:
                    print(f"[{done}/{total}] {model} ({mode}) latest={item['id']} "
                          f"acc_so_far={n_correct}/{n_graded} errors={n_err}")
            n_correct_overall += n_correct
            n_graded_overall += n_graded
            acc = n_correct / n_graded if n_graded else float("nan")
            print(f"== {model} ({mode}): accuracy={acc:.3f} ({n_correct}/{n_graded}), errors={n_err} ==")

    print(f"\nwrote {total} records -> {out_path}")
    if n_graded_overall:
        print(f"overall accuracy: {n_correct_overall / n_graded_overall:.3f} "
              f"({n_correct_overall}/{n_graded_overall})")


if __name__ == "__main__":
    main()
