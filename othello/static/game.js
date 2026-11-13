const COLOR_LABEL = { black: "黒", white: "白" };
const AI_MODEL_STORAGE_KEY = "othello_ai_model_choice";

let lastState = null;

function setBusy(busy) {
  document.body.classList.toggle("busy", busy);
}

function saveSelectedModel(value) {
  try {
    localStorage.setItem(AI_MODEL_STORAGE_KEY, value);
  } catch (e) {
    // プライベートブラウジング等でlocalStorageが使えない場合は無視する
  }
}

function loadSelectedModel() {
  try {
    return localStorage.getItem(AI_MODEL_STORAGE_KEY);
  } catch (e) {
    return null;
  }
}

async function fetchModels() {
  const select = document.getElementById("ai-model");
  select.innerHTML = '<option value="">読み込み中...</option>';
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    select.innerHTML = "";
    data.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = JSON.stringify({ name: m.name, mode: m.mode });
      opt.textContent = `${m.name} (${m.mode})`;
      select.appendChild(opt);
    });
    const saved = loadSelectedModel();
    if (saved && Array.from(select.options).some((o) => o.value === saved)) {
      select.value = saved;
    }
  } catch (e) {
    select.innerHTML = `<option value="">取得失敗: ${e.message}</option>`;
  }
}

function renderBoard(state) {
  const boardDiv = document.getElementById("board");
  boardDiv.innerHTML = "";
  if (!state.board) return;
  const legalSet = new Set(state.legal_moves || []);
  const lastMove = state.last_move;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const notation = String.fromCharCode(97 + c) + (r + 1);
      const v = state.board[r][c];
      if (v === "B" || v === "W") {
        const disc = document.createElement("div");
        disc.className = "disc " + (v === "B" ? "black" : "white");
        if (lastMove && lastMove.row === r && lastMove.col === c) {
          disc.classList.add("last-move");
        }
        cell.appendChild(disc);
      } else if (legalSet.has(notation)) {
        cell.classList.add("legal");
        cell.addEventListener("click", () => makeMove(notation));
      }
      boardDiv.appendChild(cell);
    }
  }
}

function renderStatus(state, extraNote) {
  const statusDiv = document.getElementById("status");
  const counts = state.counts || { black: 0, white: 0 };
  let text;
  if (state.status === "game_over") {
    text = `終局: 黒${counts.black} - 白${counts.white}  `;
    text += state.winner === "draw" ? "引き分け" : `${COLOR_LABEL[state.winner]}の勝ち`;
  } else if (state.status === "playing") {
    text = `現在: ${COLOR_LABEL[state.current_player]}の番 (黒${counts.black} - 白${counts.white})`;
    if (state.message) text += `  [${state.message}]`;
  } else {
    text = "モデルを選んで「新しい対局」を押してください。";
  }
  if (extraNote) text += `  ${extraNote}`;
  statusDiv.textContent = text;
}

function renderLog(state) {
  const logDiv = document.getElementById("log");
  const lines = state.log || [];
  logDiv.innerHTML = "";
  lines.forEach((line) => {
    const el = document.createElement("div");
    el.className = "log-line";
    if (line.includes("パス")) el.classList.add("pass");
    if (line.startsWith("終局")) el.classList.add("game-over");
    el.textContent = line;
    logDiv.appendChild(el);
  });
  logDiv.scrollTop = logDiv.scrollHeight;
}

function render(state, extraNote) {
  lastState = state;
  renderBoard(state);
  renderStatus(state, extraNote);
  renderLog(state);
}

function flashCopied(btn) {
  const original = btn.dataset.originalLabel || btn.textContent;
  btn.dataset.originalLabel = original;
  btn.textContent = "コピーしました";
  setTimeout(() => {
    btn.textContent = original;
  }, 1500);
}

function copyWithExecCommand(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (e) {
    ok = false;
  }
  document.body.removeChild(textarea);
  return ok;
}

async function copyLog() {
  const lines = (lastState && lastState.log) || [];
  const text = lines.join("\n");
  const btn = document.getElementById("copy-log-btn");

  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      flashCopied(btn);
      return;
    } catch (e) {
      // 権限が無い環境などでは execCommand へフォールバック
    }
  }

  if (copyWithExecCommand(text)) {
    flashCopied(btn);
  } else {
    alert("コピーに失敗しました。ログを手動で選択してコピーしてください。");
  }
}

async function makeMove(notation) {
  setBusy(true);
  try {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move: notation }),
    });
    const state = await res.json();
    if (state.error) {
      alert(state.error);
      return;
    }
    // まず自分の着手(石の反転)をすぐに画面へ反映する
    render(state);

    if (state.status === "playing") {
      render(state, "(AI思考中...)");
      const res2 = await fetch("/api/ai_turn", { method: "POST" });
      const state2 = await res2.json();
      render(state2);
    }
  } finally {
    setBusy(false);
  }
}

async function newGame() {
  const humanColor = document.getElementById("human-color").value;
  const modelSelect = document.getElementById("ai-model");
  if (!modelSelect.value) {
    alert("対戦相手モデルを選択してください。");
    return;
  }
  const modelInfo = JSON.parse(modelSelect.value);
  setBusy(true);
  try {
    const res = await fetch("/api/new_game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        human_color: humanColor,
        ai_model: modelInfo.name,
        ai_mode: modelInfo.mode,
      }),
    });
    const state = await res.json();
    if (state.error) {
      alert(state.error);
      return;
    }
    render(state);
  } finally {
    setBusy(false);
  }
}

document.getElementById("new-game-btn").addEventListener("click", newGame);
document.getElementById("refresh-models-btn").addEventListener("click", fetchModels);
document.getElementById("copy-log-btn").addEventListener("click", copyLog);
document.getElementById("ai-model").addEventListener("change", (e) => {
  saveSelectedModel(e.target.value);
});

fetchModels();
