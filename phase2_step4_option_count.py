"""Step4: choice 型質問の選択肢数(3/10/30)を変えたときの正解率とレイテンシを比較する。

対象: ticket_classification の `team` 質問(元は4択: billing/shipping/technical/other)。
追加選択肢は「紛らわしいが明確に誤りである」ことを狙って作成。
"""
import json
import os
import time

from runner import ask
from extract import extract_answer

HERE = os.path.dirname(__file__)
MODELS = ["nimble", "tev1", "tev1:0.8b"]

BASE_DESC = {
    "billing": "payment, refund, or charge issues",
    "shipping": "delivery or logistics issues",
    "technical": "product defects, bugs, or integration issues",
    "other": "anything else",
}

EXTRA_10 = {
    "sales": "pre-purchase questions about pricing or product selection",
    "security": "account compromise, suspicious login, or data breach concerns",
    "legal": "legal disputes, contracts, or compliance complaints",
    "returns_warranty": "product return or warranty claim processing (not a billing charge dispute)",
    "account_management": "changing account settings, email, or personal info",
    "marketing_feedback": "feedback about ads, promotions, or marketing emails",
}

EXTRA_30 = {
    **EXTRA_10,
    "fraud_review": "suspected fraudulent transaction requiring investigation",
    "subscription_cancellation": "request to cancel a recurring subscription",
    "product_feedback": "general feedback or suggestions about a product, not a complaint",
    "beta_program": "issues related to joining or using a beta/early-access program",
    "enterprise_support": "support requests from enterprise/business accounts specifically",
    "api_support": "issues integrating with or using a developer API",
    "data_privacy": "requests about personal data deletion, export, or privacy policy",
    "accessibility": "accessibility issues with the product or website",
    "loyalty_program": "questions about loyalty points or rewards program",
    "gift_cards": "issues specifically about gift card balance or redemption",
    "press_inquiries": "media or press requests, not customer support",
    "careers": "job application or recruiting related messages",
    "investor_relations": "investor or shareholder related inquiries",
    "international_shipping": "customs, duties, or cross-border shipping issues specifically",
    "order_modification": "request to change an order before it ships (not a delivery problem)",
    "price_matching": "request to match a lower price seen elsewhere",
    "promo_codes": "issues applying or validating a promo/discount code",
    "mobile_app_issues": "bugs specific to the mobile app, not the website or product itself",
    "website_bugs": "bugs specific to the website, not the product itself",
    "social_media_complaints": "complaints originally posted on social media and forwarded here",
}


def build_criteria(n):
    if n == 3:
        d = {k: v for k, v in BASE_DESC.items() if k != "other"}
    elif n == 10:
        d = {**BASE_DESC, **EXTRA_10}
    elif n == 30:
        d = {**BASE_DESC, **EXTRA_30}
    else:
        raise ValueError(n)
    assert len(d) == n, (n, len(d))
    return d


def main():
    items = [json.loads(l) for l in open(os.path.join(HERE, "data", "eval_set.jsonl"), encoding="utf-8")
              if json.loads(l)["task"] == "ticket_classification"]

    out_path = os.path.join(HERE, "results", "phase2", "step4_option_count_raw.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # model を一番外側のループにして、リクエストごとのモデル切り替え(ロード/アンロードの
    # 繰り返し)によるコールドスタートがレイテンシ測定に混入しないようにする。
    with open(out_path, "w", encoding="utf-8") as out_f:
        for model in MODELS:
            # ウォームアップ1回(最初のコールドロード分を切り捨てる)
            ask("http://ubuntu:11434", model, items[0]["state"], items[0]["questions"])
            for n_options in (3, 10, 30):
                criteria = build_criteria(n_options)
                q = {"type": "choice", "instructions": "How should this support ticket be routed?",
                     "criteria": criteria}
                for item in items:
                    gold = item["gold"]["team"]
                    if n_options == 3 and gold not in criteria:
                        continue  # 3択版では gold が選択肢に無いケースは対象外
                    t0 = time.monotonic()
                    result = ask("http://ubuntu:11434", model, item["state"], {"team": q})
                    latency = time.monotonic() - t0
                    choice = None
                    if result["ok"]:
                        answers, _ = extract_answer("decision", result["raw"], {"team": q})
                        choice = answers.get("team")
                    record = {"item_id": item["id"], "n_options": n_options, "model": model,
                              "ok": result["ok"], "choice": choice, "gold": gold,
                              "latency_s": latency, "error": result["error"]}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
