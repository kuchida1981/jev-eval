"""Step3-a: 同じ入力・質問セットを5回繰り返し実行し、回答の一致率と確率のばらつきを測る。"""
import json
import os
import statistics as stats

from runner import ask
from extract import extract_answer

HERE = os.path.dirname(__file__)
MODELS = ["nimble", "tev1", "tev1:0.8b"]
N_REPEATS = 5


def main():
    items = [json.loads(l) for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8")]
    out_path = os.path.join(HERE, "results", "phase2", "step3_stability_raw.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = len(items) * len(MODELS) * N_REPEATS
    done = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in items:
            for model in MODELS:
                for rep in range(N_REPEATS):
                    result = ask("http://ubuntu:11434", model, item["state"], item["questions"])
                    answers, confidences = {}, {}
                    if result["ok"]:
                        answers, confidences = extract_answer("decision", result["raw"], item["questions"])
                    record = {"item_id": item["id"], "model": model, "rep": rep, "ok": result["ok"],
                              "answers": answers, "confidences": confidences, "error": result["error"]}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done += 1
                    if done % 100 == 0 or done == total:
                        print(f"[{done}/{total}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
