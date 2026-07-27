# 原本: Policy 出力の公式 golden 照合 — 全 run が合格基準 0.05 を満たさない

- 日付: 2026-07-27
- 種別: doc（検証ログ）
- 文脈: 記事が報告する golden action MSE 0.013194（合格ライン 0.05）と同じ基準で本環境の
  Policy 出力を採点した（ユーザー指示「golden MSE の照合を先にやって」）。GPU 再実行なし、
  既存 `sample_outputs.json` の後処理のみ。

## 1. 方法

- golden: `inputs/omni/action_policy_robot.json` の `extra.golden_action_path` が指す
  `https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/bridge_20260501_0.json`
  を 2026-07-27 にダウンロード（shape [16,10]）。合格基準は同 JSON の `extra.golden_mse_max: 0.05`。
- メトリック: 公式実装 `cosmos_framework/inference/metrics.py:477 compute_action_mse`
  = 全要素の `mean((gt - pred)^2)`。同一式を Python で再実装して計算。
- 対象: `result/*/action_policy_robot/sample_outputs.json` の `outputs[0].content.action`（16×10）計 11 run。

## 2. 結果

| run | golden MSE | 判定(<0.05) |
|---|---|---|
| 記事（DGX Spark, 検証版）報告値 | 0.013194 | PASS |
| classmethod_policy_framework（6/1 相当・最適化前 1965 秒 run） | 0.126471 | FAIL |
| mainline_repro_20260726 | 0.133256 | FAIL |
| mainline_repro_v3_20260726 | 0.134059 | FAIL |
| mainline_full_v4_20260726（現行基準 run） | 0.128000 | FAIL |
| mincond_v5_20260727（42.44 秒 run） | 0.127631 | FAIL |
| control_v6_20260727 | 0.129879 | FAIL |
| policy_proposals_test / _cached / _pytorch_image | 0.128〜0.132 | FAIL |
| teacache_quality/baseline_run1・thresh_0.00 | 0.133256 | FAIL |

全 11 run が 0.126〜0.134 の狭い帯に集中。**合格ラインの約 2.6 倍、記事の約 10 倍**。

## 3. 判定の妥当性検算

- スケール一致: golden mean +0.1645 / sd 0.5802、本環境 +0.1605 / 0.5751（正規化ズレなし）
- 相関 r = 0.808（対応付けは正しい。無関係なら ~0）
- ベースライン: 全ゼロ予測 MSE 0.3637、golden 平均予測 0.3367 → 本環境 0.128 はそれより良い
  =「方向は合っているが精度が足りない」状態

## 4. 構造分析

- run 間 pairwise MSE（本環境内ノイズフロア）: **最大 0.0086**（55 ペア全列挙）。
  golden との差 0.12 はその **14 倍** → 乱数や run-to-run 変動ではなく**系統差**。
- dim 9（グリッパー、golden 値域 ±1）: MSE 0.507 で全体の 40%。時系列を見ると
  **golden は step 8 で閉じるのに対し本環境は step 6 で閉じる（2 ステップ早い）**。
  flip した step 6,7 以外の 14 ステップは ±0.015 で一致。記事の「グリッパーが開閉する瞬間だけ
  少し偏差」と失敗モードは同型で、程度が違う。
- グリッパーを除いた dims 0-8 のみでも MSE 0.084〜0.086 で**依然 FAIL**。
  グリッパータイミングだけの問題ではない。

## 5. 棄却できた原因候補（実測ベース）

1. **数値精度（fp16 説）**: 棄却。run ログ（console.log）に
   `OmniMoTModel: precision torch.bfloat16`。Policy 経路は記事と同じ BF16。
   （fp16 は diffusers 経路の話で Policy には無関係）
2. **NATTEN 不在 / sparsity 差**: 棄却。config は `joint_attn_implementation: 'two_way'` /
   `natten_parameter_list: None`。NATTEN sparsity は 'three_way' 限定
   （`cosmos3_vfm_network.py:128-130`）。SDPA fallback が置換しているのは密 attention。
3. **速度最適化の副作用**: 棄却。最適化前の 6/1 相当 run（1965 秒）が既に 0.126471 で、
   最適化後（0.127〜0.134）と同帯。**誤差は最適化導入前から一貫して存在**。
   （= 一連の高速化は品質を劣化させていない、の実証でもある）

## 6. 残る原因候補（未切り分け、本ドキュメント時点）

- SDPA varlen fallback の意味論差（causal_type 無視、GQA、softmax 精度）
- AOTriton experimental attention カーネル（`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`）の数値精度
- MIOpen conv（VAE encode）の数値差の 30 ステップ蓄積
- チェックポイント版差（記事は「正式リリース前の検証版」、golden は 2026-05-01 付）
- action 正規化統計の版差（domain registry bridge_orig_lerobot）

留意: 記事は DGX Spark（GB10）で golden（NVIDIA 生成、生成環境不明）に対し 0.013 を出している。
異なる CUDA アーキ間で 0.013 に収まるなら、正しいスタックでは軌道が数値差に対して頑健
ということであり、本環境の 0.128 は「小さな数値差のカオス増幅」より「どこかの演算の意味論
ないし精度の系統的な差」を示唆する。

## 7. 6 月からの残課題への回答

`docs/cosmos3-nano-rocm-classmethod-article-speed-benchmark.md` §今後の追加比較の
「Policy Model の action MSE を golden action と比較する後処理」→ 実施済み。結果は不合格。
