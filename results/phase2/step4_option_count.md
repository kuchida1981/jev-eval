# Step4: 選択肢数(3/10/30)バリエーション

## 手法

`ticket_classification` の `team` 質問(元は4択: billing/shipping/technical/other)を対象に、3/10/30択の criteria を作成。追加選択肢は「紛らわしいが明確に誤りである」ことを狙った(例: `returns_warranty` 「商品の返品・保証処理」、`international_shipping` 「国際配送の関税問題」など)。3択版は `other` を除外するため、gold が billing/shipping/technical の4シナリオのみを対象、10/30択版は全6シナリオを対象。

**初回計測はループ順序の不備(モデルを最内側に置いたため、リクエストごとにモデルが入れ替わりVRAM上でロード/アンロードを繰り返していた)によりレイテンシがコールドスタートで汚染されていたため、モデルを最外側ループに修正し、各モデルにウォームアップ1回を入れた上で再計測した**(汚染版は `step4_option_count_raw_confounded.jsonl` として残置、正式な結果は `step4_option_count_raw.jsonl`)。

## 結果: 正解率

| n_options | nimble | tev1 | tev1:0.8b |
|---|---|---|---|
| 3 (n=4) | 1.000 | 1.000 | 1.000 |
| 10 (n=6) | 0.833 | 0.833 | 0.667 |
| 30 (n=6) | 0.667 | 0.833 | 0.667 |

## 結果: レイテンシ(平均秒、ウォーム状態)

| n_options | nimble | tev1 | tev1:0.8b |
|---|---|---|---|
| 3 | 0.076 | 0.045 | 0.020 |
| 10 | 0.131 | 0.070 | 0.027 |
| 30 | 0.274 | 0.160 | 0.054 |
| **30/3 倍率** | **3.6x** | **3.6x** | **2.7x** |

## 誤答の内訳

| n_options | model | item | gold | 誤答 |
|---|---|---|---|---|
| 10, 30 | nimble, tev1, tev1:0.8b | ticket_classification-en-medium-004 | technical | **returns_warranty** |
| 30 | nimble | ticket_classification-en-hard-005 | other | subscription_cancellation |
| 10, 30 | tev1:0.8b | ticket_classification-ja-hard-002 | other | technical(オプション数に関係なく一貫して誤答。3択時点では該当シナリオ非対象) |

## 所見

1. **レイテンシは選択肢数にほぼ比例して増加する**: 3→30択(選択肢数10倍)でレイテンシは2.7〜3.6倍に増加した。線形よりは緩やかだが、無視できない増加であり、選択肢数の多い分類タスク(例: 数十カテゴリの商品分類)では `/v1/systemone` のレイテンシ優位性(Phase1では baseline比100〜500倍速)が目減りする可能性がある。
2. **正解率の低下は「選択肢数が増えること」自体より「特定の紛らわしい選択肢が追加されること」に強く依存する**: 3モデル共通で `ticket_classification-en-medium-004`("I don't want a refund, I just want the broken part replaced under warranty")が、`returns_warranty` という選択肢が追加された瞬間(10択・30択どちらでも)に `technical` から `returns_warranty` へ一貫して誤答(?)に転じた。ただしこれは **「誤答」というより、選択肢セット自体が変わったことで妥当な答えも変わった可能性がある**(入力に明示的に"warranty"という語があり、`returns_warranty` という専用カテゴリがあるなら、そちらの方が `technical` より適切という解釈も成立する)。元の4択データセットの gold ラベル設計が「選択肢が4つしかない前提」だったことの限界を示す副産物であり、選択肢数バリエーション実験そのものの価値(=選択肢セットの設計が結果を左右する)を裏付けている。
3. 10→30択への追加増分(26個のさらなる紛らわしい選択肢追加)自体は、nimbleで1件追加の誤答を生んだ以外は大きな影響がなかった。**「選択肢を数個追加する」ことの方が「大量に追加する」ことよりも最初のインパクトが大きい**可能性があるが、n=6と非常に小さいサンプルでの観察であり、この傾向を確証するには対象タスク・シナリオ数を増やした再検証が必要。
