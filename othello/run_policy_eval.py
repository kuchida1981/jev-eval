"""方針A(期待値最大化)と方針B(スウィンドル狙い: 劣勢時に相手のミスを誘発しやすい
手を選ぶ)を、完璧な相手(エンジン自身)と不完全な相手(Ollamaモデル)に対して
それぞれ何十局も対戦させ、勝率を比較する。

検証したい仮説(検証メモ「次回の課題案」参照):
  - 完璧な相手に対しては方針Aが方針B以上の成績を残すはず(検証可能な予測)。
  - 不完全な相手に対しては方針Bの方が勝率が高くなるかもしれない、というのが
    検証したい仮説。

「勝敗」を1局だけで判断すると運の影響が大きすぎる(このセッションで実際に
tev1:0.8bの「8連続逸脱」という単一対局の物語が、120局面の大規模検証では
再現しなかった教訓がある)ため、何十局まとめて勝率を比較する設計にしている。

使い方:
  # 完璧な相手(エンジン自身の方針A)に対して、方針A/Bそれぞれ20局
  .venv/bin/python othello/run_policy_eval.py --opponent perfect --games 20 \
      --out results/othello_policy_eval_perfect.jsonl

  # 不完全な相手(tev1:0.8b)に対して、方針A/Bそれぞれ20局
  .venv/bin/python othello/run_policy_eval.py --opponent model \
      --opponent-model tev1:0.8b --opponent-mode decision --games 20 \
      --out results/othello_policy_eval_imperfect.jsonl
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


def policy_a_move(board, player, legal):
    """方針A: エンジンの評価値(期待値)が最大の手を選ぶ(=既存のmax()と同じ)。"""
    scores, depth = search.evaluate_candidates(board, player, legal)
    best = max(scores, key=lambda mv: scores[mv])
    return best, scores, depth


def policy_b_move(board, player, legal):
    """方針B: 自分の評価値(方針A基準)上位3件に絞った上で、スウィンドル指標
    (相手が間違えた場合にどれだけ得するか)が最大の手を選ぶ。"""
    scores, depth = search.evaluate_candidates(board, player, legal)
    best, _swindle_scores = search.choose_policy_b_move(board, player, legal, scores)
    return best, scores, depth


def play_game(policy_fn, policy_name, agent_color, opponent_kind, opponent_model,
              opponent_mode, base_url, seed, random_open_plies=4, timeout=60):
    """1局プレイする。agent_color ('B' or 'W') が policy_fn を使う側。

    opponent_kind: 'perfect'(方針Aのエンジン自身) または 'model'(Ollamaモデル)。
    """
    random.seed(seed)
    board = engine.initial_board()
    player = "B"
    ply = 0
    policy_b_triggered = 0

    while True:
        b_legal = engine.legal_moves(board, "B")
        w_legal = engine.legal_moves(board, "W")
        if not b_legal and not w_legal:
            break
        legal = engine.legal_moves(board, player)
        if not legal:
            player = engine.opponent(player)
            continue

        if ply < random_open_plies:
            r, c = random.choice(list(legal.keys()))
        elif player == agent_color:
            best, scores, _depth = policy_fn(board, player, legal)
            if policy_name == "B" and max(scores.values()) < 0:
                policy_b_triggered += 1
            r, c = best
        else:
            if opponent_kind == "perfect":
                best, _scores, _depth = policy_a_move(board, player, legal)
                r, c = best
            else:
                info = ai_player.choose_move(
                    base_url, opponent_model, opponent_mode, board, player, legal, timeout=timeout
                )
                r, c = info["row"], info["col"]

        board = engine.apply_move(board, player, r, c)
        player = engine.opponent(player)
        ply += 1

    counts = engine.counts(board)
    agent_discs = counts["black"] if agent_color == "B" else counts["white"]
    opp_discs = counts["white"] if agent_color == "B" else counts["black"]
    if agent_discs > opp_discs:
        winner = "agent"
    elif opp_discs > agent_discs:
        winner = "opponent"
    else:
        winner = "draw"
    return {
        "agent_color": agent_color,
        "winner": winner,
        "agent_discs": agent_discs,
        "opponent_discs": opp_discs,
        "policy_b_triggered_count": policy_b_triggered,
    }


def print_summary(records):
    by_policy = defaultdict(list)
    for r in records:
        by_policy[r["policy"]].append(r)
    print("\n=== 集計 ===")
    for policy, rs in sorted(by_policy.items()):
        n = len(rs)
        wins = sum(1 for r in rs if r["winner"] == "agent")
        draws = sum(1 for r in rs if r["winner"] == "draw")
        losses = n - wins - draws
        avg_margin = sum(r["agent_discs"] - r["opponent_discs"] for r in rs) / n
        triggered = sum(r["policy_b_triggered_count"] for r in rs)
        print(
            f"policy {policy}: 勝 {wins}/{n}  分 {draws}  負 {losses}  "
            f"平均石差 {avg_margin:+.1f}  (方針B発動回数合計={triggered})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["A", "B", "both"], default="both")
    parser.add_argument("--opponent", choices=["perfect", "model"], required=True)
    parser.add_argument("--opponent-model", default="tev1:0.8b")
    parser.add_argument("--opponent-mode", default="decision", choices=["decision", "baseline"])
    parser.add_argument("--base-url", default="http://ubuntu:11434")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--out", default=os.path.join(HERE, "..", "results", "othello_policy_eval.jsonl")
    )
    args = parser.parse_args()

    policies = []
    if args.policy in ("A", "both"):
        policies.append(("A", policy_a_move))
    if args.policy in ("B", "both"):
        policies.append(("B", policy_b_move))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    records = []
    with open(args.out, "w", encoding="utf-8") as f:
        for policy_name, policy_fn in policies:
            label = args.opponent_model if args.opponent == "model" else "engine(policy A)"
            print(f"[policy {policy_name}] vs {args.opponent} ({label}) x {args.games} games ...")
            t0 = time.monotonic()
            for i in range(args.games):
                agent_color = "B" if i % 2 == 0 else "W"  # 先手後手を交互に
                seed = args.seed_base + i
                result = play_game(
                    policy_fn, policy_name, agent_color, args.opponent, args.opponent_model,
                    args.opponent_mode, args.base_url, seed, timeout=args.timeout,
                )
                result["policy"] = policy_name
                result["opponent"] = args.opponent
                result["opponent_model"] = args.opponent_model if args.opponent == "model" else None
                result["game_index"] = i
                result["seed"] = seed
                records.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    print(f"  {i + 1}/{args.games}  (経過 {time.monotonic() - t0:.0f}s)")
            print(f"  -> policy {policy_name} 完了 ({time.monotonic() - t0:.1f}s)")

    print_summary(records)
    print(f"\n生データ: {args.out}")


if __name__ == "__main__":
    main()
