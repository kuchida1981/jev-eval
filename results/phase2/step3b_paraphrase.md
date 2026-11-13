# Step3-b: 言い換えバリエーション測定

有効サンプル: 365 組(item×qid×model)。orig/p1/p2 を単独質問として比較。

## 全体

- n=365 flip(p1)=0.079 flip(p2)=0.088 any_flip=0.134 acc(orig/p1/p2)=0.718/0.710/0.723 McNemar p(p1)=0.710 p(p2)=0.860

## モデル別

| model | n | flip(p1) | flip(p2) | any_flip | acc orig/p1/p2 | McNemar p(p1/p2) |
|---|---|---|---|---|---|---|
| nimble | 121 | 0.074 | 0.083 | 0.116 | 0.793/0.785/0.777 | 1.000/0.752 |
| tev1 | 122 | 0.107 | 0.123 | 0.189 | 0.705/0.697/0.713 | 1.000/1.000 |
| tev1:0.8b | 122 | 0.057 | 0.057 | 0.098 | 0.656/0.648/0.680 | 1.000/0.450 |

## 質問型別

| qtype | n | flip(p1) | flip(p2) | any_flip | acc orig/p1/p2 |
|---|---|---|---|---|---|
| choice | 96 | 0.052 | 0.031 | 0.073 | 0.781/0.750/0.792 |
| noul | 125 | 0.056 | 0.072 | 0.104 | 0.808/0.816/0.784 |
| score | 144 | 0.118 | 0.139 | 0.201 | 0.597/0.590/0.625 |

## タスク別

| task | n | flip(p1) | flip(p2) | any_flip | acc orig/p1/p2 |
|---|---|---|---|---|---|
| model_routing | 36 | 0.250 | 0.083 | 0.278 | 0.611/0.528/0.583 |
| moderation | 54 | 0.037 | 0.167 | 0.185 | 0.778/0.778/0.833 |
| prompt_injection | 48 | 0.062 | 0.042 | 0.083 | 0.625/0.604/0.625 |
| rag_relevance | 48 | 0.062 | 0.062 | 0.083 | 0.771/0.792/0.833 |
| sentiment_intent | 54 | 0.074 | 0.074 | 0.111 | 0.722/0.685/0.685 |
| ticket_classification | 53 | 0.038 | 0.038 | 0.057 | 0.792/0.830/0.830 |
| tool_exec_safety | 72 | 0.083 | 0.125 | 0.167 | 0.694/0.694/0.653 |

## 言い換えで回答が変わった組み合わせ(any_flip, 49/365件中、最大15件)

| item_id | qid | model | orig | p1 | p2 | gold |
|---|---|---|---|---|---|---|
| model_routing-en-hard-017 | route | tev1:0.8b | human_escalation | small_fast | human_escalation | human_escalation |
| model_routing-en-hard-017 | complexity | tev1 | Low | Medium | Low | Medium |
| model_routing-en-medium-016 | route | nimble | large_accurate | human_escalation | large_accurate | large_accurate |
| model_routing-en-medium-016 | route | tev1 | human_escalation | human_escalation | large_accurate | large_accurate |
| model_routing-en-medium-016 | complexity | nimble | High | Medium | Medium | High |
| model_routing-ja-hard-014 | route | nimble | large_accurate | human_escalation | large_accurate | human_escalation |
| model_routing-ja-hard-014 | route | tev1 | human_escalation | small_fast | human_escalation | human_escalation |
| model_routing-ja-hard-014 | complexity | tev1 | Low | Medium | Low | Medium |
| model_routing-ja-medium-013 | complexity | nimble | High | Medium | High | High |
| model_routing-ja-medium-013 | complexity | tev1:0.8b | High | Medium | Medium | High |
| moderation-en-hard-023 | violates_policy | tev1 | True | False | True | False |
| moderation-en-hard-023 | category | tev1 | safe | harassment | harassment | safe |
| moderation-en-hard-023 | severity | tev1 | Medium | Medium | Low | Low |
| moderation-en-medium-022 | severity | tev1 | Medium | Medium | Low | Medium |
| moderation-en-medium-022 | severity | tev1:0.8b | High | High | Medium | Medium |

## 所見

1. **model_routing が言い換えでも突出して不安定**: any_flip rate 0.278(p1単体では0.250)と全タスク中で最も高く、正解率も言い換え後に悪化する方向(0.611→0.528/0.583)。Phase1 の選択肢順序バイアス追試(flip rate 0.389)と合わせて、**model_routing は「選択肢順序」「質問文の言い換え」のどちらの摂動にも弱い、プロンプトの表層変化に敏感なタスクである**ことが2系統の独立した追試で裏付けられた。criteria 具体化(Step3-c)で改善が見られたことと合わせると、曖昧な抽象カテゴリを扱う質問は、表現の具体化が最優先の対策と言える。
2. 他タスクの any_flip rate はおおむね0.06〜0.19の範囲で、moderation のみ p2(「breach typical safety or harassment guidelines」という言い換え)で突出(0.167)している。特定の言い換え表現が偶然トリガーになっている可能性があり、言い換え内容自体にも依存することに注意。
3. 正解率の変化はどのタスク・モデルでも McNemar検定で有意水準に届かない(全てp>0.24)。n=36〜72と限られたサンプルのため、「言い換えで正解率が有意に変わる」とは言えない。ただしmodel_routingだけは一貫して悪化方向に出ており、傾向としては注視すべき。
4. **手法上の注意**: 本測定は各質問を単独(他の質問と切り離して)送信しており、Phase1本集計(同一アイテムの全質問をまとめて送信)より全体的に正解率がやや低め(orig基準で0.718 vs Phase1全体0.72〜0.80台)。質問を単独で送るか・まとめて送るかという**バッチ方法自体が結果に影響しうる**ことを示唆しており、本番API設計では「複数質問をまとめて送る」運用の方が有利な可能性がある(Step4/Step5の設計にも関わる)。
