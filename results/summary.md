# Jev 代替決定モデル ローカル評価 サマリー

## 評価の概要(手法)

- **目的**: TypeSafe AI の Jev(決定モデル)のローカル代替候補として、Ollama 上で動く decision モデル(`nimble` 9B, `tev1` 4B, `tev1:0.8b` 0.8B)の精度・速度・キャリブレーションを、通常のチャットモデル(`qwen3.5:9b` + 構造化JSON出力)と比較する。
- **データ**: 人手ラベル付きの自作合成データ48シナリオ・122問。7題材(チケット分類/感情・意図/モデルルーティング/モデレーション/プロンプトインジェクション判定/ツール実行安全性/RAG関連度)× 日英半々 × 難易度3層(easy/medium/hard)。質問型は `choice`(多肢選択)/`noul`(yes-no確率)/`score`(順序尺度上の位置)の3種を混在させた。
- **難易度の設計意図**: easy = 人間が迷わない、medium = 否定表現や紛らわしい選択肢を含む、hard = 皮肉・文脈依存・情報不足などで人間でも判断が割れうる意図的に曖昧なケース。
- **実行方法**: decision モデルは Ollama の `/v1/systemone` API(確率・信頼度つきの構造化出力)、baseline は通常の `/api/chat`(`format: "json"`)でプロンプトから同じ質問に答えさせる。
- **採点**: 各質問につき、モデルの回答(`choice`はラベル、`noul`は0.5閾値の真偽、`score`は連続値を丸めて尺度ラベルに変換したもの)と gold ラベルの完全一致で正誤判定。

## 設問と回答例

### 例1: 平易なケース(easy) — `ticket_classification-ja-easy-000`

> 入力: 「商品が届きません。注文番号12345、発送から2週間経過しています。」

| qid | 型 | 選択肢/尺度 | gold | nimble の回答 |
|---|---|---|---|---|
| team | choice | billing / shipping / technical / other | shipping | shipping ✅ |
| refund_requested | noul | yes/no | false | false ✅ |
| urgency | score | Routine < Soon < Urgent | Soon | Soon ✅ |

### 例2: 否定表現(medium) — `ticket_classification-ja-medium-001`

> 入力: 「今回は返金は求めていませんが、二度目の誤配送なので改善してほしいです。」(= 返金は求めていない、という否定表現を正しく読めるかのテスト)

| qid | 型 | gold | nimble の回答 | 確信度 |
|---|---|---|---|---|
| team | choice | shipping | shipping ✅ | 0.979 |
| refund_requested | noul | **false**(返金は求めていない) | **false** ✅ | 0.999 |
| urgency | score | Soon | Routine ❌ | 0.566 |

→ 否定表現そのものは高い確信度で正しく処理できていた(キーワードの表層一致ではなく文脈を読めている可能性)。一方で urgency(緊急度)は低確信度のまま外した。

### 例3: 皮肉・文脈依存(hard) — `sentiment_intent-ja-hard-008`

> 入力: 「はいはい、最高のサービスですね(笑)。」(皮肉。字面通りに読むと誤判定する設計)

| qid | 型 | gold | nimble の回答 | 確信度 |
|---|---|---|---|---|
| intent | choice | complaint(皮肉で不満を表明) | praise ❌ | 0.991 |
| is_complaint | noul | true | false ❌ | 0.999 |
| sentiment_intensity | score | Strong | Strong ✅ | 0.758 |

→ 皮肉を字面通りに「称賛」と読み、しかも**99%以上の高確信度で**誤っている。確信度による足切り運用では救えないタイプの失敗例。

## 所感・示唆

1. **速度と精度のトレードオフが非常に良い**: `nimble` は baseline(`qwen3.5:9b`)に0.03ポイント差まで迫る精度(0.795 vs 0.827)を、中央値で約130倍速いレイテンシ(0.15秒 vs 19.15秒)で達成した。ルーティングやモデレーションのような高スループット・低レイテンシが求められる用途では、decision モデルをまず検討する価値がある。
2. **否定表現には強いが、皮肉・文脈依存には全モデルが弱い**: 例2のように「〜は求めていません」という明示的な否定は高確信度で正しく処理できる一方、例3のような皮肉は全モデルが同じ方向に、かつ高確信度で誤った。これは confidence 閾値による足切り運用では検出できない失敗モードであり、皮肉や感情的に込み入った入力が多いユースケース(SNSモデレーション等)では decision モデル単体に委ねるのは危険。
3. **`score`(順序尺度)質問が全モデル共通の弱点、特に小型モデルで顕著**: choice/noulに比べ score の正解率が低く(nimble 0.708 vs choice 0.812 / noul 0.881)、`tev1`・`tev1:0.8b`では0.521まで落ちる。urgency や severity のような連続的な強度判断は、decision モデルの出力をそのまま使うより、閾値を緩める・人間レビューを挟む設計が無難。
4. **`model_routing` タスクが全モデルで最弱(0.5〜0.67)**: `human_escalation` のような抽象的なカテゴリの判断は、criteria の記述を具体化しないと小型モデルほど不安定になりやすい。
5. **プロンプトインジェクションの確信度を系統的に低く見積もる**: gold が `High` であるべき明白な injection 事例でも、全モデルが `confidence=Medium` に留まった(B-4 の一致表参照)。セキュリティ用途で「高confidenceのみ自動ブロック」という運用をすると、見逃しが発生しうる。
6. **モデルサイズに比例してキャリブレーションが改善**: ECE は nimble(9B)0.047 < tev1(4B)0.088 < tev1:0.8b(0.8B)0.130 の順。大きい decision モデルほど「確信度=実際の正解率」の対応が良く、閾値運用(低confidenceを人間に回す)の設計がしやすい。
7. **decision API 自体の運用安定性は高い**: decision モデル3種は144リクエスト中エラー0件・レイテンシのブレも小さい。一方 baseline はthinkingの長さにより48件中7件(14.6%)が60秒タイムアウトした。レイテンシの予測可能性という観点でも decision API には明確な利点がある。

## モデル別 総合正解率・エラー率

| model | accuracy | n(graded) | errors | error rate |
|---|---|---|---|---|
| nimble | 0.795 | 97/122 | 0/48 | 0.000 |
| qwen3.5:9b | 0.827 | 86/104 | 7/48 | 0.146 |
| tev1 | 0.705 | 86/122 | 0/48 | 0.000 |
| tev1:0.8b | 0.672 | 82/122 | 0/48 | 0.000 |

## A. 難易度別正解率(easy/medium/hard)

| model | easy | hard | medium |
|---|---|---|---|
| nimble | 0.920 | 0.639 | 0.778 |
| qwen3.5:9b | 0.979 | 0.565 | 0.794 |
| tev1 | 0.900 | 0.639 | 0.500 |
| tev1:0.8b | 0.700 | 0.722 | 0.583 |

## B-1. 質問型別正解率 (choice/noul/score)

| model | choice | noul | score |
|---|---|---|---|
| nimble | 0.812 | 0.881 | 0.708 |
| qwen3.5:9b | 0.923 | 0.838 | 0.756 |
| tev1 | 0.812 | 0.833 | 0.521 |
| tev1:0.8b | 0.719 | 0.810 | 0.521 |

## B-2. 言語別正解率 (ja/en)

| model | en | ja |
|---|---|---|
| nimble | 0.803 | 0.787 |
| qwen3.5:9b | 0.827 | 0.827 |
| tev1 | 0.738 | 0.672 |
| tev1:0.8b | 0.672 | 0.672 |

## B-3. タスク別正解率

| model | model_routing | moderation | prompt_injection | rag_relevance | sentiment_intent | ticket_classification | tool_exec_safety |
|---|---|---|---|---|---|---|---|
| nimble | 0.583 | 1.000 | 0.625 | 0.688 | 0.722 | 0.889 | 0.917 |
| qwen3.5:9b | 1.000 | 0.933 | 0.750 | 0.714 | 0.778 | 0.833 | 0.857 |
| tev1 | 0.500 | 0.778 | 0.625 | 0.750 | 0.667 | 0.778 | 0.750 |
| tev1:0.8b | 0.667 | 0.667 | 0.438 | 0.938 | 0.722 | 0.778 | 0.542 |

## B-4. decision モデル間の一致度

- Fleiss' kappa: 0.657 (n=122)
- 全モデル一致・全数回答できた質問: 122
- 全モデル一致 かつ 正解: 56
- 全モデル一致 かつ 誤答(ラベル側の問題の可能性あり): 9

### 全モデルが同じ誤答をした項目(要確認)

- `sentiment_intent-ja-hard-008` / `intent`: 全モデルが `praise` と回答 (gold=`complaint`)
- `sentiment_intent-ja-hard-008` / `is_complaint`: 全モデルが `False` と回答 (gold=`True`)
- `sentiment_intent-en-medium-010` / `sentiment_intensity`: 全モデルが `Moderate` と回答 (gold=`Mild`)
- `prompt_injection-ja-easy-025` / `confidence`: 全モデルが `Medium` と回答 (gold=`High`)
- `prompt_injection-ja-medium-026` / `confidence`: 全モデルが `Medium` と回答 (gold=`High`)
- `prompt_injection-en-easy-029` / `confidence`: 全モデルが `Medium` と回答 (gold=`High`)
- `prompt_injection-en-medium-030` / `confidence`: 全モデルが `Medium` と回答 (gold=`High`)
- `tool_exec_safety-ja-hard-035` / `risk_category`: 全モデルが `destructive` と回答 (gold=`needs_confirmation`)
- `rag_relevance-en-medium-046` / `relevance`: 全モデルが `Irrelevant` と回答 (gold=`Partially relevant`)

## C. キャリブレーション(decision モデルのみ、baseline は確率非対応)

### nimble — ECE = 0.047

| bin avg confidence | bin accuracy | n |
|---|---|---|
| 0.509 | 0.542 | 24 |
| 0.729 | 0.708 | 24 |
| 0.909 | 0.875 | 24 |
| 0.985 | 0.875 | 24 |
| 0.998 | 0.958 | 24 |
| 1.000 | 1.000 | 2 |

閾値 vs 正解率/回答できる割合:
| threshold | coverage | accuracy |
|---|---|---|
| 0.0 | 1.000 | 0.795 |
| 0.3 | 0.984 | 0.808 |
| 0.5 | 0.967 | 0.814 |
| 0.6 | 0.820 | 0.860 |
| 0.7 | 0.730 | 0.888 |
| 0.8 | 0.631 | 0.896 |
| 0.9 | 0.525 | 0.906 |
| 0.95 | 0.451 | 0.891 |

### tev1 — ECE = 0.088

| bin avg confidence | bin accuracy | n |
|---|---|---|
| 0.432 | 0.417 | 24 |
| 0.650 | 0.375 | 24 |
| 0.830 | 0.875 | 24 |
| 0.952 | 0.917 | 24 |
| 0.993 | 0.917 | 24 |
| 0.999 | 1.000 | 2 |

閾値 vs 正解率/回答できる割合:
| threshold | coverage | accuracy |
|---|---|---|
| 0.0 | 1.000 | 0.705 |
| 0.3 | 0.992 | 0.711 |
| 0.5 | 0.861 | 0.752 |
| 0.6 | 0.754 | 0.804 |
| 0.7 | 0.656 | 0.875 |
| 0.8 | 0.541 | 0.924 |
| 0.9 | 0.443 | 0.926 |
| 0.95 | 0.344 | 0.905 |

### tev1:0.8b — ECE = 0.130

| bin avg confidence | bin accuracy | n |
|---|---|---|
| 0.307 | 0.375 | 24 |
| 0.514 | 0.750 | 24 |
| 0.717 | 0.542 | 24 |
| 0.869 | 0.750 | 24 |
| 0.977 | 0.917 | 24 |
| 1.000 | 1.000 | 2 |

閾値 vs 正解率/回答できる割合:
| threshold | coverage | accuracy |
|---|---|---|
| 0.0 | 1.000 | 0.672 |
| 0.3 | 0.910 | 0.703 |
| 0.5 | 0.738 | 0.767 |
| 0.6 | 0.631 | 0.740 |
| 0.7 | 0.549 | 0.776 |
| 0.8 | 0.377 | 0.848 |
| 0.9 | 0.262 | 0.875 |
| 0.95 | 0.221 | 0.889 |

## D. レイテンシ(秒)

| model | median | p95 | min | max | n |
|---|---|---|---|---|---|
| nimble | 0.15 | 0.16 | 0.10 | 0.16 | 48 |
| tev1 | 0.09 | 0.10 | 0.07 | 3.67 | 48 |
| tev1:0.8b | 0.04 | 0.04 | 0.03 | 1.12 | 48 |
| qwen3.5:9b | 19.15 | 60.01 | 2.65 | 60.01 | 48 |

## E. 追試: 選択肢順序バイアス(yes/no 反転・choice 反転)

**手法**: `choice` 型質問は criteria dict をそのまま反転。`noul` 型質問は `{"yes":.., "no":..}` の2択 `choice` に変換した上で forward=(yes,no) / reversed=(no,yes) の順で比較(ご依頼の「yes|no を no|yes へ入れ替える」に対応)。他の質問からの干渉を避けるため、各 (item, qid) を単独の質問として送信。`score` 型は尺度の意味そのものが順序のため対象外。decision モデル3種 × 74組 × forward/reversed = 444リクエスト。

| 区分 | n | flip rate | accuracy (forward / reversed) | McNemar p値 |
|---|---|---|---|---|
| 全体 | 222 | 0.059 (13/222) | 0.802 / 0.815 | 0.579(有意差なし) |
| choice 型 | 96 | 0.104 (10/96) | 0.781 / 0.823 | 0.343(有意差なし) |
| noul→yes/no 型 | 126 | 0.024 (3/126) | 0.817 / 0.810 | 1.000(有意差なし) |
| model=nimble | 74 | 0.068 (5/74) | 0.811 / 0.851 | 0.371(有意差なし) |
| model=tev1 | 74 | 0.041 (3/74) | 0.811 / 0.797 | 1.000(有意差なし) |
| model=tev1:0.8b | 74 | 0.068 (5/74) | 0.784 / 0.797 | 1.000(有意差なし) |

**タスク別 flip rate(重要な発見)**:

| task | flip rate |
|---|---|
| model_routing | **0.389 (7/18)** |
| ticket_classification | 0.056 (2/36) |
| moderation | 0.056 (2/36) |
| prompt_injection | 0.042 (1/24) |
| tool_exec_safety | 0.021 (1/48) |
| sentiment_intent | 0.000 (0/36) |
| rag_relevance | 0.000 (0/24) |

**結論**: 全体・質問型別・モデル別のどの切り口でも統計的有意差は出なかった(McNemar検定 p>0.05、ただしn=222は検出力が高くない点に注意)。一方で **`model_routing` タスクだけ突出して高い順序依存性(38.9%)** を示した。これは `model_routing` が元々全モデルで最も正解率が低いタスク(所感・示唆 4 参照)であることと整合的で、`small_fast` / `large_accurate` / `human_escalation` のように選択肢の意味が抽象的・境界が曖昧なケースほど、提示順序に引っ張られやすいと考えられる。例: `model_routing-en-hard-017` では nimble が forward=`small_fast` → reversed=`human_escalation`(gold=`human_escalation`)と、順序を変えただけで誤答から正答に変わった。本番運用でこの種のタスクを decision モデルに任せる場合は、選択肢順序をランダム化した複数回実行の多数決、または criteria の記述をより具体的にすることを推奨する。

生データ: [results/order_bias_raw.jsonl](results/order_bias_raw.jsonl)、集計: [results/order_bias_summary.json](results/order_bias_summary.json)

## 参考: 他者申告の精度(未検証、ベンダー自己申告またはサードパーティ記事)

- Nimble 75.7% / Jev 1.13 76.0% / Tev1 4B 73.3% / Tev1 0.8B 63.5% (Ollama ブログ記事記載の 13 データセット・3,880 判断の平均精度。自前計測ではない)
- 今回の自前データセット(48シナリオ)は人間でも割れる曖昧ケースを含む少数データのため、上記の大規模ベンチマークと直接比較しないこと。

## 今回のスコープで未実施(次フェーズ)

- ~~選択肢シャッフルによる順序バイアス測定~~ → 実施済み(E章参照)
- 質問の言い換え・繰り返し実行による安定性測定
- 選択肢数(3/10/50)バリエーション
- llama.cpp 経由モデル(d3-mini 等)、ホスト版 Jev API との比較
- VRAM 実測(`/api/ps` ベース)