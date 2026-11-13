"""各モデル(decision / baseline)が、固定されたOthelloエンジン(search.py)の推奨手に
どれだけ忠実に・低損失で追従できるかを、多数の局面で統計的に検証する。

これまでのセッションで繰り返し行ってきた「等価損失(equity loss)」比較を自動化・
大規模化したもの。モデル同士を対戦させるのではなく、同じ局面セットに対して各モデルが
単独で選んだ手を、こちらのエンジン自身の評価(ai_player.build_state_and_questions が
実際にモデルへ見せているのと同じ evaluate_candidates の結果)と比較する。

使い方:
  .venv/bin/python othello/run_model_eval.py \
      --decision nimble:latest tev1:latest tev1:0.8b \
      --baseline qwen3.5:9b \
      --positions 150 \
      [--base-url http://ubuntu:11434] [--seed-base 0] [--out results/othello_model_eval.jsonl]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import search  # noqa: E402
import ai_player  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_positions(n_positions, seed_base=0, samples_per_game=3):
    """ランダム自己対戦で局面を集める(これまでのセッションの等価損失検証と同じ方式)。

    こちらの探索エンジンで自己対戦させると、特に中盤の広い分岐でヒューリスティック
    探索そのものに時間がかかり局面生成が非常に遅くなるため使わない。局面の「質」は
    ここでは問題ではなく、空きマス数・分岐数が多様な局面を数多く集められればよいので、
    純粋なランダム着手による自己対戦で十分かつ高速。

    1ゲームから複数局面を全部拾うと同一ゲーム内で強く相関するため、ゲームごとに
    samples_per_game 件だけランダムに抜き出し、多数のシード(= 多数の異なる展開)に
    分散させることで局面間の独立性を確保する。
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
            if len(legal) >= 2:
                candidates_this_game.append((engine.copy_board(board), player))
            r, c = random.choice(list(legal.keys()))
            board = engine.apply_move(board, player, r, c)
            player = engine.opponent(player)

        random.shuffle(candidates_this_game)
        for pos in candidates_this_game[:samples_per_game]:
            positions.append(pos)
            if len(positions) >= n_positions:
                return positions
        seed += 1
    return positions


def evaluate_position_for_model(base_url, model, mode, board, player, legal, state, questions,
                                 best_move_notation, scores, depth, timeout):
    """1局面について、事前計算済みの state/questions/ground truth を使い回して
    モデルの選択と比較する(局面ごとの探索は呼び出し元で1回だけ行う)。"""
    fn = ai_player.ask if mode == "decision" else ai_player.ask_baseline
    t0 = time.monotonic()
    result = fn(base_url, model, state, questions, timeout=timeout)
    wall_latency = time.monotonic() - t0

    move_notation = None
    confidence = None
    if result.get("ok"):
        answers, confidences = ai_player.extract_answer(mode, result["raw"], questions)
        move_notation = answers.get("move")
        confidence = confidences.get("move")

    fallback = False
    if move_notation not in questions["move"]["criteria"]:
        move_notation = best_move_notation
        fallback = True

    chosen_r, chosen_c = engine.parse_notation(move_notation)
    chosen_score = scores.get((chosen_r, chosen_c))
    best_score = scores[max(scores, key=lambda mv: scores[mv])]

    agree = move_notation == best_move_notation or (
        chosen_score is not None and abs(chosen_score - best_score) < 1e-6
    )
    loss = 0.0 if agree else (best_score - chosen_score if chosen_score is not None else None)

    return {
        "model": model,
        "mode": mode,
        "empties": search.empty_count(board),
        "candidates": len(legal),
        "depth_label": depth,
        "best_move": best_move_notation,
        "chosen_move": move_notation,
        "agree": agree,
        "loss": loss,
        "confidence": confidence,
        "latency_s": result.get("latency_s", wall_latency),
        "fallback": fallback,
        "error": result.get("error"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", nargs="*", default=[], help="decision モードで評価するモデル名")
    parser.add_argument("--baseline", nargs="*", default=[], help="baseline モードで評価するモデル名")
    parser.add_argument("--positions", type=int, default=90, help="評価する局面数")
    parser.add_argument("--seed-base", type=int, default=0, help="局面生成の開始シード")
    parser.add_argument("--base-url", default="http://ubuntu:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "..", "results", "othello_model_eval.jsonl"),
        help="生データの出力先(JSONL)",
    )
    args = parser.parse_args()

    models = [(m, "decision") for m in args.decision] + [(m, "baseline") for m in args.baseline]
    if not models:
        print("エラー: --decision か --baseline で少なくとも1モデル指定してください", file=sys.stderr)
        sys.exit(1)

    print(f"[1/2] 局面を生成中 (n={args.positions}, seed_base={args.seed_base}) ...")
    t0 = time.monotonic()
    positions = generate_positions(args.positions, seed_base=args.seed_base)
    print(f"  -> {len(positions)}件生成 ({time.monotonic() - t0:.1f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    records = []
    t_all = time.monotonic()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (board, player) in enumerate(positions):
            legal = engine.legal_moves(board, player)
            # 局面ごとの探索(state/questions/ground truth)はここで1回だけ計算し、
            # 全モデルで使い回す。
            state, questions, best_move_notation = ai_player.build_state_and_questions(
                board, player, legal
            )
            scores, depth = search.evaluate_candidates(board, player, legal)

            for model, mode in models:
                record = evaluate_position_for_model(
                    args.base_url, model, mode, board, player, legal,
                    state, questions, best_move_notation, scores, depth, args.timeout,
                )
                record["position_index"] = i
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

            if (i + 1) % 10 == 0:
                elapsed = time.monotonic() - t_all
                print(f"  局面 {i + 1}/{len(positions)}  (経過 {elapsed:.0f}s)")
    print(f"[2/2] 全局面・全モデルの評価完了 ({time.monotonic() - t_all:.1f}s)")

    print_summary(records)
    print(f"\n生データ: {args.out}")


def print_summary(records):
    by_model = {}
    for r in records:
        key = (r["model"], r["mode"])
        by_model.setdefault(key, []).append(r)

    print("\n=== 集計 ===")
    header = f"{'モデル':<20} {'mode':<9} {'一致率':>8} {'平均損失':>10} {'平均確信度':>10} {'平均latency':>12} {'fallback率':>10} {'error率':>8}"
    print(header)
    print("-" * len(header))
    for (model, mode), rs in sorted(by_model.items()):
        n = len(rs)
        agree_n = sum(1 for r in rs if r["agree"])
        losses = [r["loss"] for r in rs if not r["agree"] and r["loss"] is not None]
        confs = [r["confidence"] for r in rs if r["confidence"] is not None]
        lats = [r["latency_s"] for r in rs if r["latency_s"] is not None]
        fallbacks = sum(1 for r in rs if r["fallback"])
        errors = sum(1 for r in rs if r["error"])
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_conf = sum(confs) / len(confs) if confs else float("nan")
        avg_lat = sum(lats) / len(lats) if lats else float("nan")
        print(
            f"{model:<20} {mode:<9} {agree_n}/{n:<6} {avg_loss:>10.2f} {avg_conf:>10.3f} "
            f"{avg_lat:>11.2f}s {fallbacks}/{n:<6} {errors}/{n}"
        )


if __name__ == "__main__":
    main()
