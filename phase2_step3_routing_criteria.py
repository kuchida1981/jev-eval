"""Step3-c: model_routing の criteria 記述を具体化し、
   順序バイアス(flip rate)と正解率が改善するかを元の criteria と比較する。
"""
import json
import os

from runner import ask
from extract import extract_answer

HERE = os.path.dirname(__file__)
MODELS = ["nimble", "tev1", "tev1:0.8b"]

ORIGINAL_CRITERIA = {
    "small_fast": "trivial, low-stakes request a small fast model can handle",
    "large_accurate": "complex reasoning that needs a larger, more careful model",
    "human_escalation": "too ambiguous or high-stakes to automate",
}

CONCRETE_CRITERIA = {
    "small_fast": ("a short, factual, single-step request with one clear correct answer "
                    "(e.g. a simple lookup or definition) that needs no multi-step reasoning"),
    "large_accurate": ("a request requiring multi-step reasoning, domain expertise, or analysis of a "
                        "long/complex document, where mistakes carry real cost"),
    "human_escalation": ("the request is missing information needed to act (e.g. 'it's broken' with no "
                          "details), or explicitly asks for a human, or the stakes are too high for any "
                          "automated answer"),
}

INSTRUCTIONS = "Which backend should handle this request?"


def make_q(criteria):
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}


def main():
    items = [json.loads(l) for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8")
              if json.loads(l)["task"] == "model_routing"]

    out_path = os.path.join(HERE, "results", "phase2", "step3_routing_criteria_raw.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in items:
            gold = item["gold"]["route"]
            for criteria_name, criteria in (("original", ORIGINAL_CRITERIA), ("concrete", CONCRETE_CRITERIA)):
                q_fwd = make_q(criteria)
                q_rev = make_q(dict(reversed(list(criteria.items()))))
                for model in MODELS:
                    for order_name, vq in (("forward", q_fwd), ("reversed", q_rev)):
                        result = ask("http://ubuntu:11434", model, item["state"], {"route": vq})
                        choice = None
                        if result["ok"]:
                            answers, _ = extract_answer("decision", result["raw"], {"route": vq})
                            choice = answers.get("route")
                        record = {"item_id": item["id"], "criteria": criteria_name, "model": model,
                                  "order": order_name, "ok": result["ok"], "choice": choice, "gold": gold,
                                  "error": result["error"]}
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
