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
