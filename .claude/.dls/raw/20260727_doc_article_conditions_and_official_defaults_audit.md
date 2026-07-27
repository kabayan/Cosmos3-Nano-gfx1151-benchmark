# 原本: 記事側比較条件の監査と公式デフォルトからの復元（guidance 不一致の発見）

- 日付: 2026-07-27
- 種別: doc（監査ログ）
- 文脈: ユーザー指示「4 モードについて (A) 記事側の比較対象が明確か推定か、(B) データ・パラメータが
  元リポジトリ由来か自作か、(C) 今日現在の README 測定値の再現性、を 1 つずつ潰す」。
  A・B の実施中に、ユーザー指摘「記事は公式リポジトリの評価スクリプトを動かしている。
  スクリプトを見れば分かることがあるのでは」を受けて公式デフォルトを調査し、guidance 不一致を発見。

## 1. 記事本文の一次取得

- URL: https://dev.classmethod.jp/articles/dgx-spark-cosmos3-omni-world-model-policy/
- 取得方法: WebFetch 2 回（要約が食い違ったため）→ **curl で原文 HTML を直接取得しテキスト化**
  （2026-07-27。playwright-cli は未インストールだった）
- 保存: スクラッチパッド article_text.txt（セッション限り。恒久保存が必要なら再取得）

### 記事に明記されている条件（原文引用）

| モード | 原文 |
|---|---|
| T2I | 「DGX Spark では 960×960 / 35 ステップ・モデル常駐後 22 秒・GPU メモリ約 30 GB という実測で」 |
| T2V | 「256p / 24 フレーム / 12 fps の軽い設定で、推論時間は 22 秒です」 |
| I2V | 「推論時間は 17 秒と text-to-video より短く」（条件数値は一切なし） |
| Policy | 「640×480 × 17 フレームの予測動画と 16 ステップ × 10 次元の action を、モデル常駐後 21 秒で出力しました」 |
| まとめ | 「**text-to-image / text-to-video / image-to-video は、35 ステップ・22 秒前後で実用品質の出力が得られた**」 |
| 環境 | 「DGX Spark（GB10 / ARM64 / 128 GB ユニファイドメモリ、CUDA 13.0、Ubuntu 24.04）」「Cosmos3-Nano（フル Omni 構成、**BF16** 約 30 GB）」 |
| 実行方法 | 「実行はいずれも**公式サンプルの JSON を指定するだけ**のシンプルな作り」「検証には**公式サンプルをそのまま使いました**」 |
| 注意 | 「本記事は**正式リリース前の検証版**での結果に基づいています」 |

**セッション前半の訂正**: WebFetch 要約 2 回はまとめ行「35 ステップ・22 秒前後」を拾わず
「T2V の steps は記事に記載なし」と誤報告した。原文直接取得で訂正。**T2V の steps=35 は記事準拠**。

### 記事が公式サンプルをそのまま使った確証

記事の Policy 節: 「公式が『これより誤差が小さければ合格』と決めているライン（**0.05**）」
「全体の誤差は **0.013194**」。この 0.05 は cosmos-framework の
`inputs/omni/action_policy_robot.json` → `extra.golden_mse_max: 0.05` と一致。
公式サンプル JSON をそのまま実行した直接証拠。

## 2. 軸 A の結論: 記事側条件の確定度

| モード | 記事に明記 | 記事に無い（＝本プロジェクトの仮定） |
|---|---|---|
| T2I | 960×960 / 35 steps / 22 秒 | プロンプト出所（ただし記事の作例が公式 t2i.json の内容と一致し、公式サンプルの公算大） |
| T2V | 256p / 24f / 12fps / 35 steps / 22 秒 | プロンプト（記事側も独自。公式 t2v サンプルは別内容） |
| I2V | 17 秒 / 「公式サンプルの条件画像」/ 35 steps | 解像度・フレーム数・fps（すべて T2V からの流用仮定） |
| Policy | 640×480×17f / 16×10 action / 21 秒 / 公式サンプルそのまま | 時間内訳（DLS-006 と同じ） |

## 3. 軸 B の結論: 本プロジェクトの入力データ出所

| モード | 入力 | 出所 | 判定 |
|---|---|---|---|
| T2I | prompt | `inputs/omni/t2i.json`（cosmos-framework） | 元リポジトリ |
| T2V | prompt | スクリプト内リテラル `ARTICLE_T2V_PROMPT` | **自作**（記事の日本語文を英訳 JSON 化。公式 t2v サンプルは "robot arm cleaning a plate" で赤キューブではないため、自作は不可避だった） |
| I2V | image/prompt | `Cosmos3-Nano-assets/assets/example_i2v_*` | 公式だが **framework の `inputs/omni/i2v.json` が指す `robot_153.jpg`（1280×720, md5 ac58d029…）とは別ファイル**（3034×1754, md5 1dd31d4d…）。記事がどちらを使ったか不明 |
| Policy | 観測動画/action/prompt | `inputs/omni/action_policy_robot.json` → cosmos-dependencies `bridge_20260501_0.*` | 完全に元リポジトリ由来。パラメータ上書きもゼロ（sample_args.json 実測で確認） |

## 4. 公式デフォルトの復元と guidance 不一致

`cosmos_framework/inference/defaults/<model_mode>/sample_args.json`（優先順位: CLI > 入力 JSON > このデフォルト。
`docs/inference.md` §Sample Arguments に明記）:

| mode | num_steps | **guidance** | shift | aspect_ratio | fps | num_frames |
|---|---|---|---|---|---|---|
| text2image | 50 | **4.0** | 3.0 | 1,1 | 24 | 1 |
| text2video | 35 | **6.0** | 10.0 | 16,9 | 24 | 189 |
| image2video | 35 | **6.0** | 10.0 | 16,9 | 24 | 189 |
| policy | 30 | **1.0** | 10.0 | 16,9 | 24 | 189 |

diffusers 側（`third_party/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`）:
`__call__` 既定 `num_inference_steps=35, guidance_scale=6.0`（L1205-1206）、
`do_classifier_free_guidance = guidance_scale != 1.0`（L1192-1193）。

CFG のコスト: framework `omni_mot_model.py` L2367-2369 のコメントに明記 —
「1 model forward when this rank skips CFG (**guidance == 1.0**) but **2 forwards otherwise**」。

本プロジェクトの実測条件（`result/verify_3modes_v3_20260726/{t2i,t2v,i2v}/summary.json` 実測確認）:
**T2I / T2V / I2V とも guidance = 1.0**。つまり公式デフォルト（4.0 / 6.0）で実行した場合の
**約半分の transformer 計算量**で測っている。Policy のみ公式デフォルトが 1.0 で一致。

guidance 1.0 の採用根拠を docs/ 全文検索したが、**根拠を記した文書は存在しない**
（初期 smoke test から無根拠に引き継がれている）。

## 5. 留保（確定していないこと）

- 記事は「正式リリース前の検証版」であり、**当時のデフォルトが現行と同一の保証はない**。
  現に記事の T2I は 35 steps で、現行 framework デフォルト（50）と食い違う
  （diffusers 既定 35 とは一致）。デフォルトが版間で動いている証拠が既にある。
- 記事の T2V 256p/24f/12fps も現行デフォルト（189f/24fps）と異なり、記事側も何かを上書きしている。
  「JSON を指定するだけ」の記述粒度では guidance の値までは確定できない。
- 決着には T2V を `--guidance 6.0` で 1 本実行し transformer_forward が約 2 倍になるかの実測が必要
  （未実施）。

## 6. 影響

- README §2 の T2I 1.23x / T2V 1.46x / I2V 1.47x は、記事側が公式デフォルト guidance で実行して
  いた場合、計算量が約半分の条件で出した倍率であり、成立しない可能性がある。
- README §2 NOTE「すべての処理で生成クオリティに関わる条件を一切変更せず、同一条件で実行」は
  guidance（生成品質に直結する条件）の不一致可能性を含む点で過大主張。
- Policy は guidance 一致のため本論点の影響を受けない（DLS-006 訂正後の対記事 1.98 倍は保持）。
