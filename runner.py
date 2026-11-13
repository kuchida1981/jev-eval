"""Jev 代替決定モデル用の統一ランナー。

ask():          Ollama の decision モデル (/v1/systemone) を叩く
ask_baseline(): 通常のチャットモデルに同じ questions を JSON 出力で問い合わせる (比較用)
"""
import json
import time
import requests

DEFAULT_TIMEOUT = 60


def ask(base_url, model, state, questions, timeout=DEFAULT_TIMEOUT):
    """POST /v1/systemone を叩き、{ok, latency_s, raw, error} を返す。

    raw の形式はモデル実装依存のため、ここでは生のレスポンスをそのまま保持し、
    値の取り出しは呼び出し側(analyze.py)の extract_answer() に任せる。
    """
    url = base_url.rstrip("/") + "/v1/systemone"
    payload = {"model": model, "state": state, "questions": questions}
    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = time.monotonic() - t0
        if resp.status_code != 200:
            return {"ok": False, "latency_s": latency, "raw": None,
                    "error": f"http {resp.status_code}: {resp.text[:500]}"}
        return {"ok": True, "latency_s": latency, "raw": resp.json(), "error": None}
    except requests.RequestException as e:
        return {"ok": False, "latency_s": time.monotonic() - t0, "raw": None, "error": str(e)}


def _question_prompt_fragment(qid, q):
    if q["type"] == "choice":
        opts = "\n".join(f"    - \"{k}\": {v}" for k, v in q["criteria"].items())
        return (f'  "{qid}" (choice, instructions: {q["instructions"]}):\n'
                f"{opts}\n"
                f'    -> respond with one of the keys above as the value for "{qid}"')
    if q["type"] == "noul":
        return (f'  "{qid}" (yes/no, instructions: {q["instructions"]}):\n'
                f'    -> respond with true or false as the value for "{qid}"')
    if q["type"] == "score":
        opts = ", ".join(f'"{c}"' for c in q["criteria"])
        return (f'  "{qid}" (ordinal score, instructions: {q["instructions"]}):\n'
                f"    -> respond with exactly one of: {opts}\n"
                f'    -> as the value for "{qid}"')
    raise ValueError(f"unknown question type {q['type']}")


def ask_baseline(base_url, model, state, questions, timeout=DEFAULT_TIMEOUT):
    """通常のチャット API (/api/chat, format=json) で同じ questions に答えさせる。"""
    fragments = "\n".join(_question_prompt_fragment(qid, q) for qid, q in questions.items())
    state_str = json.dumps(state, ensure_ascii=False)
    prompt = (
        "You are answering structured questions about the following input data.\n"
        f"Input data (JSON): {state_str}\n\n"
        "Answer ALL of the following questions. Reply with ONLY a single JSON object "
        "whose keys are exactly the question ids below, and whose values follow each "
        "question's required format. Do not add explanations.\n\n"
        f"{fragments}\n"
    )
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = time.monotonic() - t0
        if resp.status_code != 200:
            return {"ok": False, "latency_s": latency, "raw": None,
                    "error": f"http {resp.status_code}: {resp.text[:500]}"}
        body = resp.json()
        content = body.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            return {"ok": False, "latency_s": latency, "raw": body,
                    "error": f"json parse error: {e}; content={content[:300]}"}
        return {"ok": True, "latency_s": latency, "raw": {"parsed": parsed, "full": body}, "error": None}
    except requests.RequestException as e:
        return {"ok": False, "latency_s": time.monotonic() - t0, "raw": None, "error": str(e)}


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://ubuntu:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:9b"
    mode = sys.argv[3] if len(sys.argv) > 3 else "baseline"  # "decision" or "baseline"

    with open("data/eval_set.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f][:3]

    fn = ask if mode == "decision" else ask_baseline
    for item in items:
        result = fn(base_url, model, item["state"], item["questions"])
        print("---", item["id"], f"({result['latency_s']:.2f}s)", "ok" if result["ok"] else "ERROR")
        if result["ok"]:
            print(json.dumps(result["raw"], ensure_ascii=False, indent=2)[:1500])
        else:
            print("error:", result["error"])
