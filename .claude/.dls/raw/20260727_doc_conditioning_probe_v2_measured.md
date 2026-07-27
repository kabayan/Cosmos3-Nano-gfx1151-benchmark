# conditioning エンコードの実測（プローブ v2）— 無駄の確定と削減幅の測定（原本）

- 日時: 2026-07-27 00:50 〜 01:35
- 検証者: CC
- 実行: `scratchpad/probe_cond_v2_inner.py`（MIOpen ユーザー DB マウント、各エンコード 2 回、
  17 / 5 / 1 フレームの 3 通り、latent frame [0:1] で比較、sequence_plan を実測ダンプ）
- 出力: `result/conditioning_probe_v2_20260727/`
- 先行原本: `20260726_doc_conditioning_full_frame_encode_verification.md`（**§2-3 の 189 フレーム主張は誤り**）、
  `20260727_doc_conditioning_probe_refutes_189frame_claim.md`（上記の棄却と再設計）

## 1. sequence_plan の実測（静的読解の裏付け）

```json
{"sequence_plan_condition_frame_indexes_vision": [0],
 "sequence_plan_condition_frame_indexes_action": [],
 "sequence_plan_has_action": true}
```

`build_sequence_plan_from_mode`（`data/vfm/action/transforms.py` L296-299）の静的読解どおり、
policy の vision 条件付けは **latent frame [0] のみ**。action は全て生成対象（条件付けなし）。

解決済み `sample_args.json` の `condition_frame_indexes_vision: [0,1]` は CLI 既定値であり
policy 経路では消費されないことが、これで実測により確定した。

## 2. エンコード時間の実測（warm = 2 回目）

| 入力 | latent 出力 | 1 回目 | **2 回目（warm）** |
|---|---|---|---|
| 17 フレーム `[1,3,17,544,736]` | `[1,48,5,34,46]` | 713.834 s | **82.126 s** |
| 5 フレーム `[1,3,5,544,736]` | `[1,48,2,34,46]` | 1103.077 s | **214.427 s** |
| **1 フレーム** `[1,3,1,544,736]` | `[1,48,1,34,46]` | 0.209 s | **0.209 s** |

1 回目と 2 回目の差は MIOpen カーネル探索。17 フレームで 713.8 → 82.1 秒（8.7 倍）。
1 フレームだけは 1 回目から 0.209 秒で、探索が走っていない（既存の
`result/rocm_speed_matrix/miopen_user_db/gfx1151_20.HIP.*.ufdb.txt`（6/3）にヒットしたと思われる）。

**warm の 17 フレーム 82.126 秒は、mainline run の実測
`get_data_and_condition_sync` 81.575 秒（`result/mainline_full_v4_20260726/policy_stage_sync_profile.json`）と
0.7% 差で一致する。** conditioning ステージの時間は事実上すべて VAE encode である。

### 5 フレームの異常

5 フレーム（214.4 秒）が 17 フレーム（82.1 秒）より **2.6 倍遅い**。warm 同士の比較でも同じ。
エンコード時間はフレーム数に線形ではなく、形状ごとの MIOpen アルゴリズム選択に強く支配される。
前セッションの「0.43 秒/フレーム」という線形前提はここでも成立しない。
本件は削減案（1 フレーム化）には影響しないが、中間的なフレーム数を選ぶ設計は危険である。

## 3. latent frame [0] のビット一致（因果性の実証）

| 比較 | bitwise_equal | max_abs_diff |
|---|---|---|
| 17 フレーム版 vs 5 フレーム版 | **true** | 0.0 |
| 17 フレーム版 vs **1 フレーム版** | **true** | 0.0 |
| 17 フレーム版 vs 同一入力の再エンコード | true | 0.0 |

（参照値 `mean_abs_ref` = 0.4416、dtype bfloat16）

**1 ピクセルフレームだけをエンコードしても latent frame [0] はビット一致する。**
Wan2.2 VAE の因果構造（`CausalConv3d` + `feat_cache`、`wan2pt2_vae_4x16x16.py` L43-74）の
実証が取れた。同一入力の再エンコードも一致するため、非決定性も無い。

プローブ v1 で観測した `max_abs_diff: 0.03125` は latent frame **[0:2]** を比較していたためで、
policy で意味を持たない latent frame [1] の差を見ていた。

## 4. 削減幅（実測に基づく）

policy 推論において VAE encode に必要なのはピクセルフレーム 0 の 1 枚のみ。
残り 16 フレーム分の latent（latent frame 1〜4）は
`noise = cond_mask * x0 + (1 - cond_mask) * pure_noise`（`omni_mot_model.py` L1677）で
`cond_mask = 0` により捨てられ、shape だけが使われる。

| | 現状 | 1 フレーム化 |
|---|---|---|
| encode | 82.126 s | **0.209 s** |
| 削減 | — | **−81.9 s（393 倍）** |
| conditioning 込み同期総和 | 124.91 s | **約 43.0 s** |

出力は latent frame [0] がビット一致するためサンプリング入力が変わらず、**生成結果は不変**。
近似ではないため DLS-003（計算省略系を本線から除外）の対象外。

headline 41.66 秒には影響しない（`--policy-condition-cache` により conditioning は
そもそも測定対象外。DLS-005 / DLS-006）。変わるのは「conditioning 込み 124.91 秒」の側。

## 5. 実装する場合の要件（未実施）

`get_data_and_condition`（`omni_mot_model.py` L2864-2866）は
`x0_tokens_vision` を `[1,48,5,34,46]` の形で返す必要がある（ノイズ形状の決定に使われる:
L1671 `pure_noise_i = misc.arch_invariant_rand(tuple(x0_token.shape), ...)`）。
したがって「1 フレームをエンコードして latent を 5 フレーム分に埋め戻す」形になる。

未確認事項:
- `x0_tokens_vision` の他の参照箇所（L649-657 / L849 / L1368 / L1824-1861 / L2004 / L2624 / L2743）が
  推論時に latent frame 1〜4 の**値**を読んでいないか。shape のみの参照であることの確認が必要
- `raw_state_vision`（生ピクセル）は別途保持されており PSNR 検証等で使われる。こちらは削らない
- 改変対象は vendored なフレームワーク本体（`temp_src/` / `/tmp/cosmos-framework`）であり、
  本 repo のスクリプト側モンキーパッチ（`run_cosmos_framework_policy_rocm.py` の既存手法）でも実現可能

## 6. 留保

- 本測定は encode 単体。1 フレーム化した状態での**エンドツーエンドの再測定は未実施**
- 生成結果の同一性は latent frame [0] のビット一致から演繹しているが、
  実際に 1 フレーム化して action / vision.mp4 が一致することは未確認
- 各測定 1 本ずつ。ただし 17 フレーム warm が mainline 実測と 0.7% で一致しているため信頼できる
