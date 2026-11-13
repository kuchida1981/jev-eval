"""Step5: 資源計測
A) コールド/ウォームのレイテンシ差(keep_alive=0 で明示的にアンロードしてから計測)
B) 質問数 1/5/20/64 での1リクエストあたりレイテンシ
C) 同時リクエスト数 1/4/8 でのスループット
"""
import concurrent.futures
import json
import os
import time

import requests

from runner import ask

HERE = os.path.dirname(__file__)
BASE_URL = "http://ubuntu:11434"
MODELS = ["nimble", "tev1", "tev1:0.8b"]

SAMPLE_ITEM = None
for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8"):
    d = json.loads(l)
    if d["task"] == "ticket_classification":
        SAMPLE_ITEM = d
        break

TEAM_Q = SAMPLE_ITEM["questions"]["team"]


def unload(model):
    requests.post(f"{BASE_URL}/api/generate", json={"model": model, "prompt": "", "keep_alive": 0}, timeout=30)


def part_a_cold_warm():
    out = []
    for model in MODELS:
        unload(model)
        time.sleep(1)
        result_cold = ask(BASE_URL, model, SAMPLE_ITEM["state"], SAMPLE_ITEM["questions"])
        result_warm = ask(BASE_URL, model, SAMPLE_ITEM["state"], SAMPLE_ITEM["questions"])
        out.append({"model": model, "cold_latency_s": result_cold["latency_s"],
                    "warm_latency_s": result_warm["latency_s"], "cold_ok": result_cold["ok"],
                    "warm_ok": result_warm["ok"]})
        print("A:", out[-1])
    return out


def part_b_question_count():
    out = []
    counts = [1, 5, 20, 64]
    for model in MODELS:
        for n in counts:
            questions = {f"team_{i}": TEAM_Q for i in range(n)}
            result = ask(BASE_URL, model, SAMPLE_ITEM["state"], questions)
            out.append({"model": model, "n_questions": n, "latency_s": result["latency_s"],
                        "ok": result["ok"], "error": result["error"]})
            print("B:", out[-1])
    return out


def part_c_concurrency():
    out = []
    concurrencies = [1, 4, 8]
    for model in MODELS:
        for c in concurrencies:
            n_requests = c * 3
            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
                futs = [ex.submit(ask, BASE_URL, model, SAMPLE_ITEM["state"], SAMPLE_ITEM["questions"])
                        for _ in range(n_requests)]
                results = [f.result() for f in futs]
            wall = time.monotonic() - t0
            n_ok = sum(r["ok"] for r in results)
            throughput = n_ok / wall if wall > 0 else float("nan")
            out.append({"model": model, "concurrency": c, "n_requests": n_requests, "wall_s": wall,
                        "n_ok": n_ok, "throughput_rps": throughput})
            print("C:", out[-1])
    return out


def main():
    a = part_a_cold_warm()
    b = part_b_question_count()
    c = part_c_concurrency()
    out_path = os.path.join(HERE, "results", "phase2", "step5_resource_raw.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cold_warm": a, "question_count": b, "concurrency": c}, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
