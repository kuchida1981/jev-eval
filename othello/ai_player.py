"""Ollama 上のモデルにオセロの着手を選ばせる。

jev-eval 本体の runner.ask() / ask_baseline() / extract.extract_answer() をそのまま再利用する。
盤面の合法手を questions["move"] (type: choice) の criteria として渡し、モデルに
キー(例 "d3")を選ばせる方式にすることで、既存の decision モデル評価の枠組み
(nimble/tev1/tev1:0.8b は /v1/systemone、通常チャットモデルは /api/chat)と
同じ経路でオセロの一手を評価できる。
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runner import ask, ask_baseline  # noqa: E402
from extract import extract_answer  # noqa: E402

import engine  # noqa: E402
import search  # noqa: E402

_CAPS_CACHE = {}  # base_url -> {model_name: [capabilities]}


def get_model_capabilities(base_url, force_refresh=False):
    if not force_refresh and base_url in _CAPS_CACHE:
        return _CAPS_CACHE[base_url]
    import requests

    url = base_url.rstrip("/") + "/api/tags"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    caps = {m["name"]: m.get("capabilities", []) for m in data.get("models", [])}
    _CAPS_CACHE[base_url] = caps
    return caps


def mode_for_model(base_url, model):
    caps = get_model_capabilities(base_url)
    found = caps.get(model, caps.get(model + ":latest"))
    return "decision" if found and "decision" in found else "baseline"


def _render_board(board):
    return ["".join(row) for row in board]


def build_state_and_questions(board, player, legal):
    color = "black" if player == "B" else "white"
    opp_color = "white" if player == "B" else "black"
    opp = engine.opponent(player)

    scores, search_depth = search.evaluate_candidates(board, player, legal)
    # 高い評価値のものほど先頭に来るよう並べる(モデルへの提示順バイアスを避けるため
    # 強い手を先頭に固定するのではなく、あくまで判断材料として評価値自体を本文に明記する)。
    best_score = max(scores.values())

    criteria = {}
    for (r, c), flips in legal.items():
        note = engine.notation(r, c)
        score = scores[(r, c)]

        sim = engine.copy_board(board)
        engine.apply_move(sim, player, r, c)
        opp_replies = len(engine.legal_moves(sim, opp))
        weight = engine.POSITIONAL_WEIGHTS[r][c]

        tags = []
        if (r, c) in engine.CORNER_SQUARES:
            tags.append("CORNER square - extremely strong, cannot ever be flipped back")
        else:
            danger_corner = engine.DANGER_SQUARE_CORNER.get((r, c))
            if danger_corner is not None and board[danger_corner[0]][danger_corner[1]] == ".":
                tags.append(
                    "DANGER: adjacent to an empty corner - likely to hand the opponent that corner next turn"
                )
        if score == best_score:
            tags.append("BEST by lookahead search")
        tag_str = ("; " + "; ".join(tags)) if tags else ""

        if search_depth == "exact":
            eval_label = (
                "exact final disc-difference after reading to the very end of the game "
                "(no approximation at all - this is the true, provably correct result)"
            )
        else:
            eval_label = f"lookahead evaluation after {search_depth + 1}-ply best-response search "

        criteria[note] = (
            f"place {color} disc at {note} (row {r + 1}, column {chr(ord('a') + c)}); "
            f"{eval_label} "
            f"(this is the most reliable signal - higher is better for you): {score:+.0f}; "
            f"flips {len(flips)} {opp_color} disc(s); positional weight {weight:+d}; "
            f"leaves {opp_color} with {opp_replies} legal replies afterwards{tag_str}"
        )

    state = {
        "game": "othello",
        "board_rows_top_to_bottom": _render_board(board),
        "legend": {"B": "black disc", "W": "white disc", ".": "empty"},
        "columns_left_to_right": "a-h",
        "rows_top_to_bottom": "1-8",
        "player_to_move": color,
        "opponent": opp_color,
        "disc_count": engine.counts(board),
    }
    questions = {
        "move": {
            "type": "choice",
            "instructions": (
                f"You are playing Othello (Reversi) as {color}. Choose the strongest legal move from "
                "the list below.\n"
                "Each option lists a 'lookahead evaluation' computed by searching several moves ahead "
                "assuming both sides play their best response; this already accounts for consequences "
                "like losing a corner later, not just the immediate result. In general, pick the option "
                "with the highest lookahead evaluation (tagged 'BEST by lookahead search'). Do NOT simply "
                "pick the move that flips the most discs - that is a well-known beginner mistake and often "
                "has a much worse lookahead evaluation than other options.\n"
                "The remaining fields (positional weight, CORNER/DANGER tags, opponent's legal replies) are "
                "supporting context that explains *why* a move is good or bad; use them only to break ties "
                "between options whose lookahead evaluation is very close."
            ),
            "criteria": criteria,
        }
    }
    best_move_notation = engine.notation(*max(scores, key=lambda mv: scores[mv]))
    return state, questions, best_move_notation


def choose_move(base_url, model, mode, board, player, legal, timeout=60):
    """AI の一手を選ぶ。失敗時や不正な応答時は探索上のベストムーブにフォールバックする。"""
    state, questions, best_move_notation = build_state_and_questions(board, player, legal)
    fn = ask if mode == "decision" else ask_baseline
    t0 = time.monotonic()
    result = fn(base_url, model, state, questions, timeout=timeout)
    wall_latency = time.monotonic() - t0

    move_notation = None
    confidence = None
    if result.get("ok"):
        answers, confidences = extract_answer(mode, result["raw"], questions)
        move_notation = answers.get("move")
        confidence = confidences.get("move")

    fallback = False
    if move_notation not in questions["move"]["criteria"]:
        move_notation = best_move_notation
        fallback = True

    r, c = engine.parse_notation(move_notation)
    return {
        "model": model,
        "mode": mode,
        "move": move_notation,
        "row": r,
        "col": c,
        "confidence": confidence,
        "latency_s": result.get("latency_s", wall_latency),
        "fallback": fallback,
        "error": result.get("error"),
    }
