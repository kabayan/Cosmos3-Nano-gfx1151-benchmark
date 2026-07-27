# `--policy-min-condition-encode` の E2E 検証と run-to-run 非決定性フロアの測定

日付: 2026-07-27
種別: doc（実測記録）
関連: DLS-008（本フラグの実装判断）、DLS-005（Policy 再現手順）、DLS-006（対外数値）

---

## 1. 目的

DLS-008 で実装した `--policy-min-condition-encode`（conditioning のエンコード入力を
17 ピクセルフレーム → 先頭 1 フレームに削り、latent frame 1〜4 を複製で埋める）について、
エンドツーエンドで

1. conditioning 時間が実測どおり削減されるか
2. 生成結果が非適用時と変わらないか

を検証する。DLS-008 の削減効果（124.91 → 約 43 秒）はエンコード単体の実測からの推定で、
E2E では未実証だった。

## 2. run の一覧

いずれも同一コンテナ条件（`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`、framework 経路、
`--warmup-runs 2 --policy-sync-profile`）。差はフラグの有無と実行日時のみ。

| run | ディレクトリ | フラグ | 実行 |
|---|---|---|---|
| v4（基準） | `result/mainline_full_v4_20260726/` | 無し | 2026-07-26 |
| v5 | `result/mincond_v5_20260727/` | `--policy-min-condition-encode` | 07-27 05:55〜06:30 |
| v6（対照） | `result/control_v6_20260727/` | 無し | 07-27 06:35〜07:21 |

v6 は v5 と v4 の出力が不一致だったことを受け、実装由来か run-to-run 非決定性かを
切り分けるために追加した（`tasks/todo.md` Active に定めた手順）。

## 3. 速度（v4 → v5）: 削減は再現した

`policy_stage_sync_profile.json` の `summary_by_phase.measured`（秒）:

| stage | v4（基準） | v5（mincond） | 差 |
|---|---:|---:|---:|
| `get_data_and_condition_sync` | 81.5746 | **0.2764** | −81.2982 |
| `prepare_inference_data_sync` | 81.6155 | 0.2879 | −81.3276 |
| `generate_samples_from_batch_sync` | 117.2530 | 34.6022 | −82.6508 |
| `get_velocity_sync` | 35.5392 | 34.2654 | −1.2738 |
| `decode_sync` | 7.3477 | 7.5314 | +0.1837 |
| **`generate_batch_sync`** | **124.9058** | **42.4379** | **−82.4679** |

- DLS-008 の推定「約 43 秒」と一致（実測 42.44 秒）
- 削減はすべて conditioning 由来。サンプリング本体（`get_velocity_sync`）と
  デコードは素通しで、−1.27 秒 / +0.18 秒は run-to-run 変動の範囲
- profiler に `min_condition_encode_applied` が記録され、`min_condition_encode_skipped` は無し
  （＝安全弁が発動せず本経路が適用された）

## 4. 出力: bit 一致しない。ただしフラグ無し同士でも一致しない

`action_policy_robot/sample_outputs.json` の action 160 数値要素と `vision.mp4` の md5 を比較した
（比較スクリプト: scratchpad `compare_runs.py`）。

| 比較 | vision.mp4 | action 差分要素 | max_abs_diff | mean_abs_diff |
|---|---|---:|---:|---:|
| v4 vs **v6（フラグ無し同士）** | 不一致 | 160 / 160 | **0.4328** | **0.026755** |
| v4 vs v5（フラグ有無） | 不一致 | 160 / 160 | 0.1557 | 0.013901 |
| v6 vs v5 | 不一致 | 160 / 160 | 0.2771 | 0.016462 |

action 値の絶対値は max 1.0199 / mean 0.4685。

md5:
- v4 `a8b92a0ecc066a3d482d0a13c6d766a9`
- v5 `101ebf72a9039130d4269825d4084862`
- v6 `162f9fec41fe927a375b7d42e54cdc9c`

**結論**: 本環境は同一条件・同一フラグでも run-to-run で bit 再現しない。
非決定性フロア（v4 vs v6）はフラグ有無の差（v4 vs v5）より**大きい**。
したがって v5 の出力差は実装由来と識別できない。

`tasks/todo.md` に置いていた判定基準「(3) `vision.mp4` md5 と action 出力が基準と一致」は、
**フラグ無しでも満たせない基準**であり、判定条件として無効だった。

### 4.1 seed は決定的である（差の原因ではない）

ノイズは `misc.arch_invariant_rand(shape, dtype, device, seed)` で生成され、seed は
`inference.py` の `_fallback_seed` によりサンプル同一性から決定的に導かれる
（`temp_src/cosmos_framework/model/vfm/omni_mot_model.py` L1667-1678、
`temp_src/cosmos_framework/inference/inference.py` L157-170 / L1393）。
形状が同じなら初期ノイズは同一なので、出力差は seed 由来ではなく
GPU カーネルの実行時非決定性に帰属する。

## 5. 前セッション v2 プローブの欠落

`.claude/.dls/raw/20260727_doc_conditioning_probe_v2_measured.md` に対応する
`probe_cond_v2_inner.py` は docstring の変更点 4 に「cond_mask の非ゼロ latent frame
インデックスを実測ダンプする」と書いているが、実際のコードは `probe_encode` の末尾で
`ProbeDone` を送出しており、ノイズ組み立てに到達していない。
**cond_mask は測定されていない**（`result/conditioning_probe_v2_20260727/` に json 出力も無い）。

したがって DLS-008 の assumption「latent frame 1〜4 は shape のみ参照され値は読まれない」は
静的読解のみが根拠の状態だった。§6 のプローブ v3 でこれを直接測定する。

## 6. プローブ v3: サンプリング入力の直接比較

（本節は probe v3 実行後に追記する）

（↑ §6 のプレースホルダは以下の追記で埋まった。raw は追記のみのため見出しを再掲する）

### 6.1 設計

§4 で E2E 出力の bit 一致による等価性判定が不可能と分かったため、非決定性に依存しない
比較点に切り替えた。

`_prepare_inference_data`（`omni_mot_model.py` L1577-1776）の戻り値 5 番目 `initial_noise` を
比較対象にする。これは

```
noise_i = cond_mask * x0_token + (1 - cond_mask) * pure_noise_i
```

（同 L1667-1678）の結果で、DLS-008 の assumption が主張するとおりなら
`x0_tokens_vision` がサンプリングに届く唯一の経路。ここが一致すれば、以降の差は
サンプリング側の非決定性だけになる。

同一プロセス内で 3 通り実行した:

- **B**: 最小エンコード（runner の `--policy-min-condition-encode` 実装をそのまま使用）
- **A**: 通常エンコード（runner のラッパーを外し真の `get_data_and_condition` を呼ぶ）
- **C**: 通常エンコードをもう一度（プロセス内決定性の土台確認）

スクリプト: scratchpad `probe_noise_v3_inner.py` / `run_probe_noise_v3.py`
（`MIOPEN_USER_DB_PATH` マウント付き、`--warmup-runs 0`、サンプリング前に中断）。
出力: `result/conditioning_probe_v3_20260727/probe_v3.json`

### 6.2 結果

| 比較 | bitwise_equal | max_abs_diff |
|---|---|---:|
| `initial_noise` A vs C（通常同士） | **true** | 0.0 |
| `initial_noise` A vs B（通常 vs 最小） | **true** | **0.0** |
| `x0_tokens_vision` A vs B | false | 4.96875 |

`x0_tokens_vision` の形状は `[1, 48, 5, 30, 40]`（latent frame 5 枚）。frame ごと:

| latent frame | bitwise_equal | max_abs_diff | mean_abs（通常側） |
|---|---|---:|---:|
| 0 | **true** | 0.0 | 0.4489 |
| 1 | false | 4.3037 | 0.6024 |
| 2 | false | 4.1797 | 0.6987 |
| 3 | false | 4.0781 | 0.7248 |
| 4 | false | 4.9688 | 0.7382 |

### 6.3 読み方

1. **A vs C がビット一致** → 同一プロセス内でこの経路は決定的。比較の土台が成立する
   （§4 の非決定性はサンプリング以降で発生していることも意味する）
2. **latent frame 1〜4 は実際に大きく違う値**（max_abs_diff 4.08〜4.97。値の平均絶対値
   0.60〜0.74 に対して桁が大きく、「たまたま近い」ではない）
3. **にもかかわらず `initial_noise` はビット完全一致**（max_abs_diff 0.0）

→ latent frame 1〜4 の値はサンプリング入力に一切影響しない。DLS-008 の assumption
「shape のみ参照され値は読まれない」は静的読解ではなく**実測で確定**した。
`--policy-min-condition-encode` は近似ではなく厳密であり、DLS-003 の計算省略系には
該当しない（省略しているのは結果に到達しない計算のみ）。

## 7. 結論

- 速度: `generate_batch` 124.906 → **42.438 秒**（−82.47 秒）。DLS-008 の推定と一致
- 等価性: サンプリング入力レベルで**ビット一致**を実証。厳密削減である
- E2E 出力差（v4 vs v5）は全額 GPU run-to-run 非決定性に帰属する。
  非決定性フロア（v4 vs v6、フラグ無し同士）の方が大きい
- 判定基準の教訓: **bit 一致を判定条件に置く前に、対照 run で非決定性フロアを測る**。
  フロアが不明なまま「出力一致」を基準にすると、達成不可能な基準で実装を疑うことになる
- 未解決: 対外文書（README §2 / `docs/cosmos3_rocm_policy_optimization_final_report.md`）の
  「conditioning 込み 124.91 秒」をどう扱うか。DLS-006 で参考値として併記した数値であり、
  対外主張に関わるためユーザー判断を要する
