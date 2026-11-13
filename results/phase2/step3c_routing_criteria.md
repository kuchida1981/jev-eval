# Step3-c: model_routing の criteria 具体化テスト

## 手法

`route` 質問(choice型)の criteria を、抽象的な元の記述から、判断基準を明示した具体的な記述に書き換え、model_routing の6シナリオ全件に対して forward/reversed(選択肢順序反転)の両方で実行(3モデル × 2順序 = 18組、concreteは1件APIエラーでn=17)。

**元の criteria(original)**:
- small_fast: "trivial, low-stakes request a small fast model can handle"
- large_accurate: "complex reasoning that needs a larger, more careful model"
- human_escalation: "too ambiguous or high-stakes to automate"

**具体化した criteria(concrete)**:
- small_fast: "a short, factual, single-step request with one clear correct answer (e.g. a simple lookup or definition) that needs no multi-step reasoning"
- large_accurate: "a request requiring multi-step reasoning, domain expertise, or analysis of a long/complex document, where mistakes carry real cost"
- human_escalation: "the request is missing information needed to act (e.g. 'it's broken' with no details), or explicitly asks for a human, or the stakes are too high for any automated answer"

## 結果

| criteria | n | flip rate(95%CI) | accuracy forward(95%CI) | accuracy reversed(95%CI) | McNemar p |
|---|---|---|---|---|---|
| original | 18 | 0.389 [0.203, 0.614] | 0.611 [0.386, 0.797] | 0.889 [0.672, 0.969] | 0.131(有意差なし) |
| concrete | 17 | 0.118 [0.033, 0.343] | 0.882 [0.657, 0.967] | 1.000 [0.816, 1.000] | 0.480(有意差なし) |

## 所見

1. criteria を具体化すると、flip rate は 0.389→0.118、forward順での正解率は 0.611→0.882 と、点推定では大きく改善した。ただし n=17〜18 と非常に小さく、95%信頼区間は広く重なっており(例: original forward acc CI上限0.797 と concrete forward acc CI下限0.657)、この差が偶然のばらつきでないと断定はできない。
2. McNemar検定はどちらの条件でも有意水準に届かない(p=0.13, p=0.48)。これは主に n が小さいことによる検出力不足であり、「改善していない」ことを意味しない。傾向としては一貫して改善方向に出ている。
3. **実務的な示唆**: 少なくとも悪化はしておらず、flip rate・正解率ともに改善方向の一貫した傾向が見られたことから、model_routing のような抽象カテゴリ選択では criteria の記述を具体化するコストに見合う価値がある可能性が高い。ただし本追試のサンプル数(6シナリオ)では統計的に確証できないため、本番導入前により大きなデータセット(最低でも各カテゴリ20件以上)での再検証を推奨する。
