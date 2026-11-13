"""着手ごとの「方針実行精度」(Decision能力)を測定する。

方針A(期待値最大化)・方針B(スウィンドル狙い)それぞれについて、数式的に定義した
「正解」(= search.py が計算する argmax)を用意し、同じ局面・同じ方針を自然言語で
指示したときに、各モデルの選択がその数式的な正解と一致するかを測定する。

方針Bは「劣勢時に相手のミスを誘発する」という意味を持つため、対象局面は
方針Aの評価値が負(エンジンが「劣勢」と判定した局面)に限定する。

使い方:
  .venv/bin/python othello/run_policy_match_eval.py \
      --decision nimble:latest tev1:latest tev1:0.8b \
      --positions 60 --out results/othello_policy_match_eval.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import search  # noqa: E402
import ai_player  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_losing_positions(n_positions, seed_base=0, samples_per_game=2):
    """ランダム自己対戦で、方針Aの評価値が負(「劣勢」)の局面を集める。
    方針Bはこの状況でのみ意味を持つため。

    「劣勢かどうか」の判定には、全候補手のnegamax探索(evaluate_candidates)ではなく
    軽量な static_eval(現局面だけを見る、探索なしのO(1)評価)を使う。局面生成の
    段階では大まかな目安で十分であり、実際の厳密な評価値(ground truth)は
    呼び出し側が位置ごとに evaluate_candidates で別途計算する。
    (過去に run_model_eval.py で全手に探索を使って生成が致命的に遅くなった
    教訓を踏まえた設計。)
    """
    positions = []
    seed = seed_base
    while len(positions) < n_positions:
        random.seed(seed)
        board = engine.initial_board()
        player = "B"
        candidates_this_game = []
        while True:
            b_legal = engine.legal_moves(board, "B")
            w_legal = engine.legal_moves(board, "W")
            if not b_legal and not w_legal:
                break
            legal = engine.legal_moves(board, player)
            if not legal:
                player = engine.opponent(player)
                continue
            if len(legal) >= 2 and search.static_eval(board, player) < -80:
                candidates_this_game.append((engine.copy_board(board), player))
            r, c = random.choice(list(legal.keys()))
            board = engine.apply_move(board, player, r, c)
            player = engine.opponent(player)

        random.shuffle(candidates_this_game)
        accepted_this_game = 0
        for cand_board, cand_player in candidates_this_game:
            if accepted_this_game >= samples_per_game:
                break
            # static_eval(高速だが粗い)だけでは3割程度が実は劣勢でなかったため、
            # 採用前に厳密な evaluate_candidates(どのみちこの後の評価で必要になる
            # 計算)で検証し、本当に劣勢(評価値<0)の局面だけを残す。
            cand_legal = engine.legal_moves(cand_board, cand_player)
            scores, _depth = search.evaluate_candidates(cand_board, cand_player, cand_legal)
            if max(scores.values()) >= 0:
                continue
            positions.append((cand_board, cand_player))
            accepted_this_game += 1
            if len(positions) >= n_positions:
                return positions
        seed += 1
    return positions


def build_policy_b_state_and_questions(board, player, legal, scores):
    """方針B(スウィンドル狙い)用の state/questions を構築する。
    方針Aと同じ構造(choice型、criteriaに数値根拠を明記)を踏襲し、
    一致率の測定を「評価値の見せ方」ではなく「方針の違い」に絞れるようにする。
    """
    color = "black" if player == "B" else "white"
    opp_color = "white" if player == "B" else "black"

    best_mv, swindle_scores = search.choose_policy_b_move(board, player, legal, scores)
    best_note = engine.notation(*best_mv)
    best_swindle = max(swindle_scores.values())

    criteria = {}
    for note_mv in sorted(swindle_scores, key=lambda mv: -swindle_scores[mv]):
        note = engine.notation(*note_mv)
        tag = " [BEST by swindle potential]" if note_mv == best_mv else ""
        criteria[note] = (
            f"place {color} disc at {note}; swindle potential (how much you gain if the "
            f"{opp_color} opponent fails to find their best reply, compared to if they play "
            f"perfectly): {swindle_scores[note_mv]:+.1f}{tag}"
        )
    # スウィンドル指標を計算した候補(上位数件)以外も選択肢として提示する必要があるため、
    # 残りは方針Aの評価値のみで補足する(スウィンドル指標は上位プール内でのみ計算)。
    for mv in legal:
        note = engine.notation(*mv)
        if note not in criteria:
            criteria[note] = (
                f"place {color} disc at {note}; this move was not among the top equity "
                f"candidates, so its swindle potential was not computed (likely not worth considering)"
            )

    state = {
        "game": "othello",
        "board_rows_top_to_bottom": ["".join(row) for row in board],
        "legend": {"B": "black disc", "W": "white disc", ".": "empty"},
        "columns_left_to_right": "a-h",
        "rows_top_to_bottom": "1-8",
        "player_to_move": color,
        "opponent": opp_color,
        "disc_count": engine.counts(board),
        "situation": f"{color} is currently BEHIND according to the engine's own equity evaluation",
    }
    questions = {
        "move": {
            "type": "choice",
            "instructions": (
                f"You are playing Othello (Reversi) as {color}, and you are currently BEHIND. "
                "Playing the pure equity-maximizing move is unlikely to be enough to win if the "
                f"{opp_color} opponent plays perfectly from here. Instead, adopt a swindle-seeking "
                "policy: among reasonable moves, prefer the one with the highest 'swindle potential' "
                "- a move where the opponent's perfect reply is only slightly better for them than "
                "their other replies, so that if they fail to find the single correct reply, you "
                "gain a lot. Pick the option with the highest swindle potential "
                "(tagged 'BEST by swindle potential')."
            ),
            "criteria": criteria,
        }
    }
    return state, questions, best_note, best_swindle


def evaluate_position_for_model(base_url, model, mode, board, player, legal, timeout):
    """1つの「劣勢」局面について、方針A・方針Bそれぞれで、モデルの選択が
    数式的な正解(argmax)と一致するかを測定する。"""
    scores, depth = search.evaluate_candidates(board, player, legal)

    # 方針A(既存の build_state_and_questions をそのまま利用)
    state_a, questions_a, best_a_note = ai_player.build_state_and_questions(board, player, legal)
    fn = ai_player.ask if mode == "decision" else ai_player.ask_baseline

    result_a = fn(base_url, model, state_a, questions_a, timeout=timeout)
    chosen_a = None
    if result_a.get("ok"):
        answers, _conf = ai_player.extract_answer(mode, result_a["raw"], questions_a)
        chosen_a = answers.get("move")
    agree_a = chosen_a == best_a_note

    # 方針B(スウィンドル狙い)
    state_b, questions_b, best_b_note, best_swindle = build_policy_b_state_and_questions(
        board, player, legal, scores
    )
    result_b = fn(base_url, model, state_b, questions_b, timeout=timeout)
    chosen_b = None
    if result_b.get("ok"):
        answers, _conf = ai_player.extract_answer(mode, result_b["raw"], questions_b)
        chosen_b = answers.get("move")
    agree_b = chosen_b == best_b_note

    return {
        "model": model,
        "mode": mode,
        "empties": search.empty_count(board),
        "candidates": len(legal),
        "engine_best_equity": max(scores.values()),
        "policy_a_best_move": best_a_note,
        "policy_a_chosen_move": chosen_a,
        "policy_a_agree": agree_a,
        "policy_a_error": result_a.get("error"),
        "policy_b_best_move": best_b_note,
        "policy_b_best_swindle_score": best_swindle,
        "policy_b_chosen_move": chosen_b,
        "policy_b_agree": agree_b,
        "policy_b_error": result_b.get("error"),
    }


def print_summary(records):
    by_model = defaultdict(list)
    for r in records:
        by_model[(r["model"], r["mode"])].append(r)

    print("\n=== 集計(方針実行精度) ===")
    header = f"{'モデル':<20} {'mode':<9} {'方針A一致率':>12} {'方針B一致率':>12}"
    print(header)
    print("-" * len(header))
    for (model, mode), rs in sorted(by_model.items()):
        n = len(rs)
        agree_a = sum(1 for r in rs if r["policy_a_agree"])
        agree_b = sum(1 for r in rs if r["policy_b_agree"])
        print(f"{model:<20} {mode:<9} {agree_a}/{n:<9} {agree_b}/{n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", nargs="*", default=[])
    parser.add_argument("--baseline", nargs="*", default=[])
    parser.add_argument("--positions", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--base-url", default="http://ubuntu:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--out", default=os.path.join(HERE, "..", "results", "othello_policy_match_eval.jsonl")
    )
    args = parser.parse_args()

    models = [(m, "decision") for m in args.decision] + [(m, "baseline") for m in args.baseline]
    if not models:
        print("エラー: --decision か --baseline で少なくとも1モデル指定してください", file=sys.stderr)
        sys.exit(1)

    print(f"[1/2] 「劣勢」局面を生成中 (n={args.positions}) ...")
    t0 = time.monotonic()
    positions = generate_losing_positions(args.positions, seed_base=args.seed_base)
    print(f"  -> {len(positions)}件生成 ({time.monotonic() - t0:.1f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    records = []
    t_all = time.monotonic()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (board, player) in enumerate(positions):
            legal = engine.legal_moves(board, player)
            for model, mode in models:
                record = evaluate_position_for_model(
                    args.base_url, model, mode, board, player, legal, args.timeout
                )
                record["position_index"] = i
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            if (i + 1) % 10 == 0:
                print(f"  局面 {i + 1}/{len(positions)}  (経過 {time.monotonic() - t_all:.0f}s)")
    print(f"[2/2] 全局面・全モデルの評価完了 ({time.monotonic() - t_all:.1f}s)")

    print_summary(records)
    print(f"\n生データ: {args.out}")


if __name__ == "__main__":
    main()
