# jev-eval

TypeSafe AI の Jev(決定モデル)のローカル代替候補として、Ollama 上で動く decision モデル
(`nimble` 9B, `tev1` 4B, `tev1:0.8b` 0.8B)と、通常のチャットモデル + 構造化JSON出力
(`qwen3.5:9b` 等)を比較評価するプロジェクト。

## 構成

### 1. 構造化QA評価(本体)

人手ラベル付きの合成データ(チケット分類・感情/意図・モデルルーティング・モデレーション・
プロンプトインジェクション判定・ツール実行安全性・RAG関連度、日英×easy/medium/hard)に対して、
各モデルの精度・レイテンシ・キャリブレーションを評価する。

- `runner.py` — Ollama の decision API(`/v1/systemone`)と baseline チャット API(`/api/chat`)
  を呼ぶ共通ランナー
- `extract.py` — 生レスポンスから回答値を取り出す(choice/noul/score の3型に対応)
- `run_eval.py` — 全モデル×全評価項目を実行し `results/raw_*.jsonl` に保存
- `analyze.py` / `analyze_*.py` — 集計・分析スクリプト群
- `data/eval_set.jsonl` — 評価データセット
- `results/summary.md` — 評価結果のサマリー(主要な知見はここに記載)

```bash
.venv/bin/python run_eval.py --base-url http://ubuntu:11434 \
    --decision nimble tev1 tev1:0.8b --baseline qwen3.5:9b \
    --out results/raw_full.jsonl
```

### 2. オセロベンチマーク(`othello/`)

同じモデル群を、静的な一問一答QAとは異なる「連続的・累積的な意思決定」の場で評価するための
追加実験。コード側のミニマックス探索(+終盤完全読み切り+確定石判定)が各候補手を評価し、
モデルには「評価値が最も高い選択肢を選ぶ」というシンプルな分類タスクとして着手を選ばせる。

- `othello/app.py` — プレイヤー vs AI で実際に対戦できる Flask アプリ
  (`.venv/bin/python othello/app.py` → http://127.0.0.1:5050)
- `othello/engine.py` / `othello/search.py` — オセロのルールと評価エンジン
- `othello/ai_player.py` — 盤面を `runner.ask`/`ask_baseline` の入力に変換し、AIに着手を選ばせる
- `othello/run_model_eval.py` — 複数モデルを同一の局面セットに対して自動評価し、
  エンジンの推奨手との一致率・損失量を統計的に集計するスクリプト

```bash
.venv/bin/python othello/run_model_eval.py \
    --decision nimble:latest tev1:latest tev1:0.8b \
    --baseline qwen3.5:9b \
    --positions 120 --out results/othello_model_eval.jsonl
```

詳しくは `othello/README.md` を参照。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install requests flask
```

Ollama サーバー(デフォルト `http://ubuntu:11434`)に評価対象のモデルが存在している必要がある。

## 主な知見(詳細は `results/summary.md`)

- decision モデルは baseline チャットモデルに僅差の精度で、約130倍速いレイテンシを達成。
- 否定表現には強いが、皮肉・文脈依存の入力には全モデルが弱い。
- モデルサイズが大きいほどキャリブレーション(確信度と実際の正解率の対応)が良い。
- オセロベンチマークでは、モデルの強さの差が「戦略理解」ではなく「提示された評価値への
  追従性・キャリブレーション」として現れることを確認(小さいモデルほど、選択肢が増える
  局面で推奨から逸脱しやすい)。
