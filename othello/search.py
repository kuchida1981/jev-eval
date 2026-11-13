"""オセロの数手先ミニマックス(negamax + alpha-beta)評価。

Ollama の decision/baseline モデルは一手先の情報(ひっくり返る枚数や位置重み)だけでは
人間に勝てるほど強くならなかったため、各候補手についてここで数手先まで読んだ評価値を
計算し、ai_player.build_state_and_questions() がその値を criteria の説明文として
モデルに渡す。最終的な一手の選択はモデル自身に委ねたまま、判断材料を強化する。
"""
import engine

NEG_INF = float("-inf")
POS_INF = float("inf")


MOBILITY_WEIGHT = 9
# 確定石(今後絶対にひっくり返されない石)1枚あたりの重み。位置重みの合計だけでは
# 「良さそうなマスに居るが実は不安定」な石を過大評価してしまう(石数40-10でも
# ヒューリスティックが+400の白優勢と誤判定した事例で発覚)ため導入。等価損失法で
# 0/10/20/30/40/60を比較したところ30が最も損失が小さかった(平均4.43→3.85枚)
# ため採用(詳細は検証メモ参照)。
STABILITY_WEIGHT = 30


def static_eval(board, player):
    """位置重み(角重視)+ 着手可能数の差 + 確定石の差、player 視点のスコア。"""
    opp = engine.opponent(player)
    score = 0
    for r in range(8):
        for c in range(8):
            cell = board[r][c]
            if cell == player:
                score += engine.POSITIONAL_WEIGHTS[r][c]
            elif cell == opp:
                score -= engine.POSITIONAL_WEIGHTS[r][c]
    my_mobility = len(engine.legal_moves(board, player))
    opp_mobility = len(engine.legal_moves(board, opp))
    mobility_score = 0.0
    if my_mobility + opp_mobility:
        mobility_score = 100.0 * (my_mobility - opp_mobility) / (my_mobility + opp_mobility)
    stable = engine.stable_disc_counts(board)
    stability_score = stable[player] - stable[opp]
    return score + MOBILITY_WEIGHT * mobility_score + STABILITY_WEIGHT * stability_score


def negamax(board, player, depth, alpha=NEG_INF, beta=POS_INF):
    """player 視点の評価値。両者とも最善を尽くす前提で depth 手先まで読む。"""
    opp = engine.opponent(player)
    legal = engine.legal_moves(board, player)

    if depth == 0 or (not legal and not engine.legal_moves(board, opp)):
        if not legal and not engine.legal_moves(board, opp):
            c = engine.counts(board)
            mine = c["black"] if player == "B" else c["white"]
            theirs = c["white"] if player == "B" else c["black"]
            return (mine - theirs) * 1000  # 終局は石差を最優先
        return static_eval(board, player)

    if not legal:
        return -negamax(board, opp, depth - 1, -beta, -alpha)

    # 位置重みの高いマス(角など)から先に調べると alpha-beta の枝刈りがよく効く。
    ordered = sorted(legal, key=lambda rc: engine.POSITIONAL_WEIGHTS[rc[0]][rc[1]], reverse=True)

    best = NEG_INF
    for (r, c) in ordered:
        sim = engine.copy_board(board)
        engine.apply_move(sim, player, r, c)
        val = -negamax(sim, opp, depth - 1, -beta, -alpha)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def evaluate_move(board, player, r, c, depth=4):
    """player が (r, c) に打った直後の局面を、相手視点から depth 手読んで評価し、
    player 視点のスコア(高いほど player に有利)として返す。"""
    opp = engine.opponent(player)
    sim = engine.copy_board(board)
    engine.apply_move(sim, player, r, c)
    return -negamax(sim, opp, depth, NEG_INF, POS_INF)


def _depth_for_branching(num_candidates):
    """候補手数(盤面の分岐)に応じた探索深さ。実測ベースでどの分岐数でも 1 手あたり
    おおむね 3 秒以内に収まるよう調整した固定テーブル(探索中の 1 手を時間切れで
    中断する手段がないため、時間予算より先に深さを決め打ちする方が安全で安定する)。"""
    if num_candidates <= 6:
        return 5
    if num_candidates <= 13:
        return 4
    return 3


# 終盤、空きマスがこの数以下になったら、ヒューリスティックではなく石差を完全に
# 読み切った厳密値で評価する(近似誤差ゼロ)。トランスポジションテーブル導入後の
# 実測(各空きマス数で1局面あたり全候補手を評価する合計時間、n=8):
#   空き10: 平均0.28s / 最大0.57s
#   空き11: 平均0.83s / 最大1.92s
#   空き12: 平均2.79s / 最大6.13s (ばらつきが大きく、素朴な着手順序づけの限界)
# 安全マージンを見て 11 に設定。さらに広げるには、簡易な位置重みソートではなく
# 空きマスの配置を考慮した着手順序づけや、Othello 専用のビットボード実装が必要。
EXACT_SOLVE_EMPTY_THRESHOLD = 11


def empty_count(board):
    return sum(row.count(".") for row in board)


def _board_key(board, player):
    return (tuple(cell for row in board for cell in row), player)


def exact_solve(board, player, alpha=NEG_INF, beta=POS_INF, tt=None):
    """ゲーム終了まで完全に読み切り、player視点の最終石差(自分 - 相手)を返す。
    ヒューリスティックを一切使わない、終盤専用の厳密解。

    tt はトランスポジションテーブル({局面キー: (フラグ, 値)})。別の指し手経由で
    同じ局面に到達した際の再計算を避けるため、evaluate_candidates から呼ぶ際は
    候補手すべてで共有する。フラグは fail-soft alpha-beta の標準的な分類
    (EXACT=真の値, LOWER=少なくともこの値以上, UPPER=高々この値)で、プルーニングで
    打ち切られた探索の結果を安全に再利用できるようにしている。
    """
    if tt is None:
        tt = {}

    key = _board_key(board, player)
    entry = tt.get(key)
    if entry is not None:
        flag, value = entry
        if flag == "EXACT":
            return value
        if flag == "LOWER" and value > alpha:
            alpha = value
        elif flag == "UPPER" and value < beta:
            beta = value
        if alpha >= beta:
            return value

    opp = engine.opponent(player)
    legal = engine.legal_moves(board, player)
    original_alpha = alpha

    if not legal:
        opp_legal = engine.legal_moves(board, opp)
        if not opp_legal:
            c = engine.counts(board)
            mine = c["black"] if player == "B" else c["white"]
            theirs = c["white"] if player == "B" else c["black"]
            best = mine - theirs
            tt[key] = ("EXACT", best)
            return best
        # 強制パス: 子ノードが1つだけの探索として、以下の分岐ロジックと同様に扱う。
        best = -exact_solve(board, opp, -beta, -alpha, tt)
    else:
        ordered = sorted(legal, key=lambda rc: engine.POSITIONAL_WEIGHTS[rc[0]][rc[1]], reverse=True)
        best = NEG_INF
        for (r, c) in ordered:
            sim = engine.copy_board(board)
            engine.apply_move(sim, player, r, c)
            val = -exact_solve(sim, opp, -beta, -alpha, tt)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

    if best <= original_alpha:
        flag = "UPPER"
    elif best >= beta:
        flag = "LOWER"
    else:
        flag = "EXACT"
    tt[key] = (flag, best)
    return best


def evaluate_move_exact(board, player, r, c, tt=None):
    opp = engine.opponent(player)
    sim = engine.copy_board(board)
    engine.apply_move(sim, player, r, c)
    return -exact_solve(sim, opp, tt=tt)


def evaluate_candidates(board, player, legal, use_exact_endgame=True):
    """全候補手を評価する。

    終盤(空きマスが EXACT_SOLVE_EMPTY_THRESHOLD 以下)では厳密な石差読み切りを行い、
    それ以外では盤面の分岐数に応じた深さのヒューリスティック探索を行う。

    戻り値: ({(r, c): score}, 実際に使った深さ(int) または "exact")
    """
    if use_exact_endgame and empty_count(board) <= EXACT_SOLVE_EMPTY_THRESHOLD:
        tt = {}
        scores = {move: evaluate_move_exact(board, player, move[0], move[1], tt=tt) for move in legal}
        return scores, "exact"
    depth = _depth_for_branching(len(legal))
    scores = {move: evaluate_move(board, player, move[0], move[1], depth=depth) for move in legal}
    return scores, depth


# ---------------------------------------------------------------------------
# 方針B: 「スウィンドル狙い」(劣勢時に相手のミスを誘発しやすい手を選ぶ)
#
# 「勝敗だけを正解基準にすると常に期待値最大化(方針A)に収束してしまう」という
# 問題を避けるため、方針Bは「相手が完璧に指せば方針Aと大差ないが、相手が間違えると
# 大きく得をする」手を選ぶ。これは相手が不完全な場合にのみ意味を持つ、検証可能な
# 仮説になっている(詳しくは検証メモの「次回の課題案」節を参照)。
# ---------------------------------------------------------------------------

SWINDLE_DEPTH = 2  # 相手の応手を評価する際の先読み深さ(速度とのトレードオフ)
SWINDLE_TOP_K = 3  # 自分の評価値(方針A基準)上位何件に絞ってから選ぶか


def opponent_reply_scores(board, player, r, c, depth=SWINDLE_DEPTH):
    """player が (r, c) に打った直後の局面で、相手が取りうる全応手それぞれについて、
    その応手の後の player 視点のスコアを返す(リストの各要素が相手の1つの応手に対応)。"""
    opp = engine.opponent(player)
    sim = engine.copy_board(board)
    engine.apply_move(sim, player, r, c)
    opp_legal = engine.legal_moves(sim, opp)
    if not opp_legal:
        # 相手は合法手なし(パス): 自分の番が続くとみなして評価する。
        return [negamax(sim, player, depth, NEG_INF, POS_INF)]
    scores = []
    for (rr, cc) in opp_legal:
        sim2 = engine.copy_board(sim)
        engine.apply_move(sim2, opp, rr, cc)
        scores.append(negamax(sim2, player, depth, NEG_INF, POS_INF))
    return scores


def evaluate_move_swindle(board, player, r, c, depth=SWINDLE_DEPTH):
    """手 (r, c) の「スウィンドル指標」: 相手の応手候補全体の平均的な結果と、
    相手が最善を尽くした場合(=方針Aの評価値と同義)の結果との差。

    値が大きいほど、「相手が完璧なら方針Aと大差ないが、相手が最善を外すと
    こちらが大きく得をする」手であることを示す。相手が常に最善を尽くす場合は
    この差はゼロに近づくため、方針Bは完璧な相手には方針Aを上回れない設計になっている。
    """
    reply_scores = opponent_reply_scores(board, player, r, c, depth=depth)
    worst_for_us = min(reply_scores)
    mean_for_us = sum(reply_scores) / len(reply_scores)
    return mean_for_us - worst_for_us


def choose_policy_b_move(board, player, legal, equity_scores, top_k=SWINDLE_TOP_K, depth=SWINDLE_DEPTH):
    """方針B(スウィンドル狙い)の着手を選ぶ。

    方針A(evaluate_candidatesのargmax)による評価値で上位 top_k 件に候補を絞り、
    その中からスウィンドル指標が最大の手を選ぶ。「明らかな悪手は避けつつ、相手の
    ミスを誘発しやすい手を優先する」という設計。

    戻り値: (選んだ手 (r, c), {手: スウィンドル指標} のうち計算した分)
    """
    ranked = sorted(legal, key=lambda mv: equity_scores[mv], reverse=True)
    pool = ranked[: min(top_k, len(ranked))]
    swindle_scores = {mv: evaluate_move_swindle(board, player, mv[0], mv[1], depth=depth) for mv in pool}
    best_mv = max(pool, key=lambda mv: swindle_scores[mv])
    return best_mv, swindle_scores
