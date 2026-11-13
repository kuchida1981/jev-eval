"""オセロ(リバーシ)のルール実装。盤面は 8x8 の list[list[str]]、値は 'B'/'W'/'.'。"""

SIZE = 8
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# 定石の静的評価テーブル(角は最重要、角の隣接マス(X/C-square)は危険、辺は安全寄り)。
POSITIONAL_WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

CORNER_SQUARES = {(0, 0), (0, 7), (7, 0), (7, 7)}

# X-square / C-square -> 隣接する角の座標。その角がまだ空いているときにここへ打つと
# 相手に角を取られやすくなる、という古典的な危険マス。
DANGER_SQUARE_CORNER = {
    (0, 1): (0, 0), (1, 0): (0, 0), (1, 1): (0, 0),
    (0, 6): (0, 7), (1, 7): (0, 7), (1, 6): (0, 7),
    (7, 1): (7, 0), (6, 0): (7, 0), (6, 1): (7, 0),
    (7, 6): (7, 7), (6, 7): (7, 7), (6, 6): (7, 7),
}


def copy_board(board):
    return [row[:] for row in board]


def initial_board():
    board = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    board[3][3] = "W"
    board[3][4] = "B"
    board[4][3] = "B"
    board[4][4] = "W"
    return board


def opponent(player):
    return "W" if player == "B" else "B"


def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def flips_for_move(board, player, r, c):
    if board[r][c] != ".":
        return []
    opp = opponent(player)
    flips = []
    for dr, dc in DIRS:
        line = []
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc) and board[rr][cc] == opp:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and in_bounds(rr, cc) and board[rr][cc] == player:
            flips.extend(line)
    return flips


def legal_moves(board, player):
    """{(r, c): [(r, c), ...]} 合法手 -> その手でひっくり返る石の座標リスト。"""
    moves = {}
    for r in range(SIZE):
        for c in range(SIZE):
            flips = flips_for_move(board, player, r, c)
            if flips:
                moves[(r, c)] = flips
    return moves


def apply_move(board, player, r, c):
    flips = flips_for_move(board, player, r, c)
    if not flips:
        raise ValueError(f"illegal move for {player}: {notation(r, c)}")
    board[r][c] = player
    for rr, cc in flips:
        board[rr][cc] = player
    return board


def counts(board):
    flat = [cell for row in board for cell in row]
    return {"black": flat.count("B"), "white": flat.count("W")}


_STABILITY_AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def _axis_immune(board, r, c, dr, dc):
    """ある軸(dr, dc とその逆方向)について、その石が将来ひっくり返され得ないかを判定する。

    ある方向にその石をひっくり返すには、空きマス(新しい着手の置き場)が軸の片側に、
    同色の確定した石(挟み込みの終端)がもう片側になければならない。したがって、
    軸のどちらか一方が盤外で尽きている(そちら側にマスが無い)なら、その側を起点/終点
    とする挟み込みは原理的に起こり得ず、軸全体として永久に安全となる(角・辺の石が
    周囲の空きマスに関わらず最初から安定なのはこのため)。盤外で尽きる側が無い場合は、
    従来通り「軸全体(両方向とも)に空きマスが一つも無い」ことを安全の条件とする。
    """
    rr, cc = r + dr, c + dc
    pos_len = 0
    pos_full = True
    while in_bounds(rr, cc):
        pos_len += 1
        if board[rr][cc] == ".":
            pos_full = False
            break
        rr += dr
        cc += dc

    rr, cc = r - dr, c - dc
    neg_len = 0
    neg_full = True
    while in_bounds(rr, cc):
        neg_len += 1
        if board[rr][cc] == ".":
            neg_full = False
            break
        rr -= dr
        cc -= dc

    if pos_len == 0 or neg_len == 0:
        return True
    return pos_full and neg_full


def is_stable_disc(board, r, c):
    """その石が、以降どんな着手が続いても絶対にひっくり返されない(確定石)かを判定する。

    横・縦・斜め2方向の4軸すべてで _axis_immune が成り立てば確定石とみなす。
    (保守的な簡易実装: 既に確定した隣接石からの伝播によるより緩い安定性までは
    判定しない。見逃し方向のみで、確定石でない石を確定石と誤判定することはない。)
    """
    return all(_axis_immune(board, r, c, dr, dc) for dr, dc in _STABILITY_AXES)


def stable_disc_counts(board):
    """{"B": n, "W": n} 確定石の数。"""
    result = {"B": 0, "W": 0}
    for r in range(SIZE):
        for c in range(SIZE):
            cell = board[r][c]
            if cell != "." and is_stable_disc(board, r, c):
                result[cell] += 1
    return result


def notation(r, c):
    return f"{chr(ord('a') + c)}{r + 1}"


def parse_notation(s):
    s = s.strip().lower()
    col = ord(s[0]) - ord("a")
    row = int(s[1:]) - 1
    if not in_bounds(row, col):
        raise ValueError(f"invalid notation: {s}")
    return row, col
