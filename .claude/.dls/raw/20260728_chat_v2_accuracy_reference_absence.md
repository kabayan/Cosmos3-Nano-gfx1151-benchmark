# 議論ノート: 公開 v2 checkpoint に policy 精度の基準値が存在しない

- 日付: 2026-07-28
- 種別: chat（`/dls-continue` セッション中の対話から抽出）
- 関連 DLS: DLS-011, DLS-013, DLS-014

## 出発点

E4 の実行待ち中、ユーザーの指摘:
「新しいモデルでの PASS（≈0.013）のような記述はリポジトリにないか？
ないとすると v2 での精度目安はない？」

それまでの議論は暗黙に「0.013194 は達成可能な基準値」という前提で進んでいた。
この指摘はその前提自体を問うもので、前提が崩れれば「本環境は不合格」という
問題設定そのものが変わる。

## 調査した 3 点と結果

### 1. リポジトリ内の 0.013 はすべて記事の引用（一次情報は 1 つだけ）

`0.013` の全出現を走査した結果、`scripts/check_policy_golden_mse.py` の
コメントと docs / DLS 原本の記述はすべて記事値 0.013194 の引用で、出所は単一。
DLS-013 の時系列調査によれば記事の検証実施は 2026-06-01 の super-squash 以前で、
v1 時代に跨がる可能性がある（＝そもそも v2 の値ではない可能性）。

なお E4（DLS-014）の結果、v1 重みでも 0.248 で 0.013 には遠く、
「記事値は v1 由来」という説明も成り立たなくなった。

### 2. 閾値は宣言されているが、誰も執行していない

- `inputs/omni/*.json` の `extra.golden_mse_max = 0.05` / `golden_psnr_min = 14.0`
  は **データとして書かれているだけ**。framework 全体を grep しても
  これらを読む Python コードが 1 行も存在しない
- `tests/nano_inference_smoke_test.py` 冒頭に
  "Smoke-level only (output validity, not numeric goldens)" と明記
- 数値 golden を持つ `tests/launch_regression_test.py` は **学習時の loss /
  grad-norm** を h100 / gb200 で照合するもので、公開 checkpoint の推論出力とは無関係

### 3. 公式ベンチマークに policy 指標が無い

HF `nvidia/Cosmos3-Nano` の README のベンチマーク節（画像 4 枚）のうち
action 系は `images/benchmark-action-1.png` の 1 枚だけで、内容は
**逆動力学（ID）と順動力学（FD）のみ**（RRE / RTE / ATE と PSNR）。
policy（行動生成）の精度指標は 1 つも載っていない。技術レポートも
「base model の評価」と断っている。

## 確定事項

1. **公開 v2 checkpoint が policy として golden 基準を満たすかは、NVIDIA 自身も
   第三者も測定していない**。閾値だけが宣言されて放置されている
2. v2 について世に存在する唯一の policy 精度実測は本環境の 12 run
   （golden MSE 0.126〜0.134、全 FAIL）であり、その意味で本環境の測定は
   「再現失敗」ではなく「初回測定」に近い
3. これは DLS-013 assumption の後半（「公開 v2 の golden 合格は誰も検証して
   いない可能性が高い」confidence: medium）を補強する。前半（記事は v1 で
   検証した）は E4 で棄却されたが、後半は independent に強まった

## この議論が変えたこと

「本環境が基準を満たさない」という問題設定を、
「基準を満たす実測がそもそも公開されていない」に置き換えた。
これにより README の対外表現の訂正方向も変わりうる
（「本環境が劣る」ではなく「公式基準に対する実測が本環境にしかない」）。

## 議論再開時の起点

- CUDA 参照 run が FAIL（≈0.13）だった場合、「公開資産では記事値は再現しない」
  が確定し、記事値の位置づけ（検証版限定の値だった可能性）を対外文書でどう扱うか
  が次の論点になる
- 逆に PASS なら本環境固有の問題が残ることになり、DLS-012 で棄却した精度説の
  外側（意味論差・入力側）を洗い直す必要がある

## 検証根拠（再現手段）

- `grep -rn "0\.013" --include=*.md --include=*.py --include=*.json`
- `grep -rn "golden_mse_max\|golden_action" /tmp/cosmos-framework --include=*.py`（0 件）
- `head tests/nano_inference_smoke_test.py` / `tests/launch_regression_test.py`
- HF README `## Benchmarks` 節と `images/benchmark-action-1.png`（ローカル snapshot
  411f42a8 に存在）
