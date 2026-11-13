"""Step3-b: 質問文(instructions)の言い換え2パターンで回答が変わる割合を測る。

各 (item, qid) について、元の instructions / paraphrase1 / paraphrase2 を
単独の質問として送信し(他の質問からの干渉を避ける)、回答が変わるかを見る。
"""
import copy
import json
import os

from runner import ask
from extract import extract_answer

HERE = os.path.dirname(__file__)
MODELS = ["nimble", "tev1", "tev1:0.8b"]

PARAPHRASES = {
    ("ticket_classification", "team"): [
        "Which team is best suited to handle this ticket?",
        "Decide which team should take ownership of this support request.",
    ],
    ("ticket_classification", "refund_requested"): [
        "Did the customer clearly ask for their money back?",
        "Has the customer explicitly made a refund request?",
    ],
    ("ticket_classification", "urgency"): [
        "How time-sensitive is this ticket?",
        "Rate how quickly this ticket needs a response.",
    ],
    ("sentiment_intent", "intent"): [
        "What is the main purpose behind this message?",
        "Identify the primary communicative goal of this message.",
    ],
    ("sentiment_intent", "is_complaint"): [
        "Is the customer expressing a grievance?",
        "Does this message voice a complaint?",
    ],
    ("sentiment_intent", "sentiment_intensity"): [
        "How intense is the emotion conveyed in this message?",
        "Rate the strength of the feeling expressed here.",
    ],
    ("model_routing", "route"): [
        "Which processing tier is appropriate for this request?",
        "Decide which backend system should process this request.",
    ],
    ("model_routing", "complexity"): [
        "How difficult is this request to handle?",
        "Rate the complexity level of this request.",
    ],
    ("moderation", "violates_policy"): [
        "Would this content be flagged under a standard harassment/safety policy?",
        "Does this content breach typical safety or harassment guidelines?",
    ],
    ("moderation", "category"): [
        "Which violation category best describes this content, or 'safe' if none applies?",
        "Classify this content into the most fitting policy category (or 'safe').",
    ],
    ("moderation", "severity"): [
        "Rate how severe this content is.",
        "How serious is the policy issue in this content?",
    ],
    ("prompt_injection", "is_injection"): [
        "Does this input try to bypass the AI's instructions or leak system/data information?",
        "Is the user attempting to manipulate the AI into ignoring its rules or revealing protected data?",
    ],
    ("prompt_injection", "confidence"): [
        "How severe is this prompt injection attempt likely to be?",
        "Rate your confidence that this constitutes a serious injection attempt.",
    ],
    ("tool_exec_safety", "safe_to_run"): [
        "Can this command/script be run immediately, without needing extra confirmation?",
        "Is it okay to execute this action as-is without asking the user first?",
    ],
    ("tool_exec_safety", "risk_category"): [
        "How would you classify the risk level of this command?",
        "Categorize the risk posed by this command.",
    ],
    ("tool_exec_safety", "destructiveness"): [
        "Rate how destructive this action would be.",
        "How much potential for damage does this action have?",
    ],
    ("rag_relevance", "relevance"): [
        "To what extent does this passage help answer the query?",
        "Rate how useful this passage is for answering the question.",
    ],
    ("rag_relevance", "sufficient"): [
        "Could someone answer the query using only this passage?",
        "Does this passage by itself fully answer the query?",
    ],
}


def main():
    items = [json.loads(l) for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8")]
    out_path = os.path.join(HERE, "results", "phase2", "step3_paraphrase_raw.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    jobs = []
    for item in items:
        for qid, q in item["questions"].items():
            paras = PARAPHRASES.get((item["task"], qid))
            if not paras:
                continue
            variants = {"orig": q}
            for i, p in enumerate(paras, 1):
                vq = copy.deepcopy(q)
                vq["instructions"] = p
                variants[f"p{i}"] = vq
            jobs.append((item, qid, variants))

    total = len(jobs) * len(MODELS) * 3  # orig + p1 + p2
    done = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for item, qid, variants in jobs:
            for model in MODELS:
                for version, vq in variants.items():
                    result = ask("http://ubuntu:11434", model, item["state"], {qid: vq})
                    choice = None
                    if result["ok"]:
                        answers, _ = extract_answer("decision", result["raw"], {qid: vq})
                        choice = answers.get(qid)
                    record = {"item_id": item["id"], "qid": qid, "model": model, "version": version,
                              "ok": result["ok"], "choice": choice, "gold": item["gold"].get(qid),
                              "error": result["error"]}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done += 1
                    if done % 100 == 0 or done == total:
                        print(f"[{done}/{total}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
