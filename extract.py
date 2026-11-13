"""/v1/systemone および baseline の生レスポンスから回答値を取り出す。

decision モデル (nimble/tev1 系, Ollama 0.40.14) の実レスポンス形式(2026-11-12, tev1:0.8b で確認済み):

{
  "answers": {
    "<choice_qid>": {"type": "choice", "choice": "<label>",
                      "probabilities": {"<label>": p, ...}, "confidence": c},
    "<noul_qid>":   {"type": "noul", "noul": p_yes},
    "<score_qid>":  {"type": "score", "score": <float position>,
                      "legend": {"0": "...", "1": "...", ...},
                      "probabilities": {"0": p, "1": p, ...}, "confidence": c}
  },
  "usage": {"input_tokens": n, "output_tokens": n}
}

score は離散ラベルに対応する連続値の「期待位置」(例 1.24)であり、argmax(probabilities) とは
一致しない場合がある(確率分布の期待値 vs 最頻値)。ここでは round() して legend に通す方式を採用。
"""


def _coerce_noul(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v >= 0.5
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "はい", "y")
    return None


def _from_decision_raw(raw, questions):
    """/v1/systemone の生レスポンスから {qid: value} と {qid: confidence} を抽出する。"""
    answers = {}
    confidences = {}
    container = raw.get("answers", {}) if isinstance(raw, dict) else {}

    for qid, q in questions.items():
        entry = container.get(qid)
        value, conf = None, None

        if isinstance(entry, dict):
            if q["type"] == "choice":
                value = entry.get("choice")
                probs = entry.get("probabilities", {})
                conf = probs.get(value, entry.get("confidence"))
            elif q["type"] == "noul":
                p_yes = entry.get("noul")
                if p_yes is not None:
                    value = p_yes >= 0.5
                    conf = p_yes if value else 1 - p_yes
            elif q["type"] == "score":
                score = entry.get("score")
                legend = entry.get("legend", {})
                probs = entry.get("probabilities", {})
                if score is not None and legend:
                    n = len(legend)
                    idx = max(0, min(n - 1, round(score)))
                    value = legend.get(str(idx))
                    conf = probs.get(str(idx), entry.get("confidence"))

        answers[qid] = value
        confidences[qid] = conf

    return answers, confidences


def _from_baseline_raw(raw, questions):
    parsed = raw.get("parsed", {}) if isinstance(raw, dict) else {}
    answers = {}
    for qid, q in questions.items():
        v = parsed.get(qid)
        if q["type"] == "noul":
            v = _coerce_noul(v)
        answers[qid] = v
    return answers, {qid: None for qid in questions}  # baseline は確率を返さない


def extract_answer(mode, raw, questions):
    """mode: 'decision' か 'baseline'。戻り値: (answers: {qid: value}, confidences: {qid: float|None})"""
    if raw is None:
        return {qid: None for qid in questions}, {qid: None for qid in questions}
    if mode == "decision":
        return _from_decision_raw(raw, questions)
    return _from_baseline_raw(raw, questions)


def grade(answers, gold):
    """{qid: bool} の正誤マップを返す(gold に存在する質問のみ)。"""
    return {qid: (answers.get(qid) == gold_v) for qid, gold_v in gold.items()}
