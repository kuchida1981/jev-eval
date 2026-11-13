"""Step1 追加: qwen3.5:9b がタイムアウトした7件を timeout=180秒で再実行する。"""
import json
import os
import time

from runner import ask_baseline

HERE = os.path.dirname(__file__)
FAILED_IDS = [
    "model_routing-ja-hard-014", "model_routing-ja-medium-013", "moderation-en-hard-023",
    "rag_relevance-ja-hard-043", "ticket_classification-en-easy-003",
    "ticket_classification-en-hard-005", "tool_exec_safety-ja-hard-035",
]


def main():
    items_by_id = {}
    with open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            items_by_id[d["id"]] = d

    out_path = os.path.join(HERE, "results", "phase2", "step1_baseline_180s.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for item_id in FAILED_IDS:
            item = items_by_id[item_id]
            t0 = time.monotonic()
            result = ask_baseline("http://ubuntu:11434", "qwen3.5:9b", item["state"], item["questions"],
                                   timeout=180)
            dt = time.monotonic() - t0
            print(f"{item_id}: ok={result['ok']} latency={dt:.1f}s error={result['error']}")
            record = {"item_id": item_id, "model": "qwen3.5:9b", "mode": "baseline",
                      "ok": result["ok"], "latency_s": result["latency_s"], "error": result["error"],
                      "raw": result["raw"], "gold": item["gold"], "task": item["task"],
                      "lang": item["lang"], "difficulty": item["difficulty"]}
            if result["ok"]:
                parsed = result["raw"].get("parsed", {})
                record["answers"] = parsed
                record["correct"] = {qid: (parsed.get(qid) == g) for qid, g in item["gold"].items()}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
