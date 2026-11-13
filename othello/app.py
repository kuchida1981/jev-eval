"""プレイヤー vs Ollama AI のオセロ。Flask のシングルゲーム状態をサーバー側で持つ。

起動: .venv/bin/python othello/app.py  (jev-eval/jev-eval を cwd にして)
    -> http://127.0.0.1:5050 をブラウザで開く
"""
import os

from flask import Flask, jsonify, render_template, request

import ai_player
import engine

app = Flask(__name__)

GAME = {
    "board": None,
    "current_player": None,
    "human_color": "B",
    "ai_color": "W",
    "ai_model": None,
    "ai_mode": "decision",
    "base_url": "http://ubuntu:11434",
    "status": "idle",  # idle | playing | game_over
    "message": "",
    "last_ai_move": None,
    "last_move": None,
    "winner": None,
    "log": [],
}


def _side_label(player):
    return "黒" if player == "B" else "白"


def _log(msg):
    GAME["log"].append(msg)


def _finish_if_over():
    board = GAME["board"]
    if not engine.legal_moves(board, "B") and not engine.legal_moves(board, "W"):
        c = engine.counts(board)
        GAME["status"] = "game_over"
        if c["black"] > c["white"]:
            GAME["winner"] = "black"
        elif c["white"] > c["black"]:
            GAME["winner"] = "white"
        else:
            GAME["winner"] = "draw"
        result = "引き分け" if GAME["winner"] == "draw" else f"{_side_label({'black': 'B', 'white': 'W'}[GAME['winner']])}の勝ち"
        _log(f"終局: 黒{c['black']} - 白{c['white']}  {result}")
        return True
    return False


def _advance():
    """人間の番になるか終局するまで、AI の着手とパスを自動で進める。"""
    while GAME["status"] == "playing":
        if _finish_if_over():
            break
        board = GAME["board"]
        player = GAME["current_player"]
        legal = engine.legal_moves(board, player)
        if not legal:
            _log(f"{_side_label(player)}はパス(合法手なし)")
            GAME["current_player"] = engine.opponent(player)
            continue
        if player == GAME["human_color"]:
            GAME["message"] = ""
            break
        info = ai_player.choose_move(
            GAME["base_url"], GAME["ai_model"], GAME["ai_mode"], board, player, legal
        )
        engine.apply_move(board, player, info["row"], info["col"])
        GAME["last_move"] = {"row": info["row"], "col": info["col"]}
        GAME["last_ai_move"] = {
            "move": info["move"],
            "latency_s": info["latency_s"],
            "confidence": info["confidence"],
            "fallback": info["fallback"],
            "error": info["error"],
        }
        conf_str = f"{info['confidence']:.3f}" if isinstance(info["confidence"], (int, float)) else "N/A"
        note = f"{_side_label(player)}(AI {GAME['ai_model']}): {info['move']}  latency={info['latency_s']:.2f}s confidence={conf_str}"
        if info["fallback"]:
            note += "  ※応答不正のため探索上のベストムーブにフォールバック"
        if info["error"]:
            note += f"  error={info['error']}"
        _log(note)
        GAME["current_player"] = engine.opponent(player)


def serialize_state():
    board = GAME["board"]
    player = GAME["current_player"]
    is_human_turn = GAME["status"] == "playing" and player == GAME["human_color"]
    legal = engine.legal_moves(board, player) if (board and is_human_turn) else {}
    return {
        "board": board,
        "current_player": {"B": "black", "W": "white"}.get(player),
        "human_color": {"B": "black", "W": "white"}.get(GAME["human_color"]),
        "legal_moves": [engine.notation(r, c) for (r, c) in legal],
        "counts": engine.counts(board) if board else None,
        "status": GAME["status"],
        "message": GAME["message"],
        "last_ai_move": GAME["last_ai_move"],
        "last_move": GAME["last_move"],
        "winner": GAME["winner"],
        "log": GAME["log"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    base_url = request.args.get("base_url", GAME["base_url"])
    try:
        caps = ai_player.get_model_capabilities(base_url, force_refresh=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    models = [
        {"name": name, "mode": "decision" if "decision" in c else "baseline"}
        for name, c in caps.items()
        if "decision" in c or "completion" in c
    ]
    models.sort(key=lambda m: (m["mode"], m["name"]))
    return jsonify({"models": models})


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(force=True) or {}
    human_color = data.get("human_color", "black")
    ai_model = data.get("ai_model")
    ai_mode = data.get("ai_mode", "decision")
    base_url = data.get("base_url") or GAME["base_url"]
    if not ai_model:
        return jsonify({"error": "ai_model is required"}), 400

    GAME["board"] = engine.initial_board()
    GAME["human_color"] = "B" if human_color == "black" else "W"
    GAME["ai_color"] = engine.opponent(GAME["human_color"])
    GAME["ai_model"] = ai_model
    GAME["ai_mode"] = ai_mode
    GAME["base_url"] = base_url
    GAME["current_player"] = "B"
    GAME["status"] = "playing"
    GAME["message"] = ""
    GAME["last_ai_move"] = None
    GAME["last_move"] = None
    GAME["winner"] = None
    GAME["log"] = [
        f"新しい対局開始: あなたは{_side_label(GAME['human_color'])}、相手は{ai_model}({ai_mode})"
    ]
    _advance()
    return jsonify(serialize_state())


@app.route("/api/move", methods=["POST"])
def api_move():
    if GAME["status"] != "playing":
        return jsonify({"error": "no active game"}), 400
    data = request.get_json(force=True) or {}
    note = data.get("move")
    if not note:
        return jsonify({"error": "move is required"}), 400
    player = GAME["current_player"]
    if player != GAME["human_color"]:
        return jsonify({"error": "not your turn"}), 400
    try:
        r, c = engine.parse_notation(note)
    except Exception:
        return jsonify({"error": "invalid move notation"}), 400
    legal = engine.legal_moves(GAME["board"], player)
    if (r, c) not in legal:
        return jsonify({"error": "illegal move"}), 400
    engine.apply_move(GAME["board"], player, r, c)
    GAME["last_move"] = {"row": r, "col": c}
    _log(f"{_side_label(player)}: {note}")
    GAME["current_player"] = engine.opponent(player)
    # ここでは AI の着手は進めない。プレイヤー自身の着手(石の反転)をすぐに画面へ
    # 反映できるよう、AI の手番は /api/ai_turn で別リクエストとして進める。
    _finish_if_over()
    return jsonify(serialize_state())


@app.route("/api/ai_turn", methods=["POST"])
def api_ai_turn():
    _advance()
    return jsonify(serialize_state())


@app.route("/api/state")
def api_state():
    return jsonify(serialize_state())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="127.0.0.1", port=port, debug=True)
