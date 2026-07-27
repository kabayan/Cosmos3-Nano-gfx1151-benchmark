# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-27 08:05
> 前セッションの要約: `--policy-min-condition-encode`（DLS-008）の E2E 検証を完了。速度削減は再現し
> （`generate_batch` 124.91 → 42.44 秒）、等価性はサンプリング入力のビット一致で実証した（DLS-009）。
> 対外文書（README / 最終報告）を 42.44 秒に更新済み。バックグラウンド run は残っていない。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** conditioning 最適化の一連の作業は完了・コミット済み。

**ブランチは 2 系統のまま**:
- `main`（チェックアウト中）: `5ad9927` が最新。`origin/main` より **6 コミット先行（未 push）**
- `experiment/teacache-quality-eval`: TeaCache 評価ハーネス（`3d3b514`）と DLS-004。**未マージ**

### 対外数値の現状（すべて検証済み）

| モード | 公表値 | 実測 | 差 |
|---|---|---|---|
| T2I | 27.136 s | 27.214 s | +0.3% |
| T2V | 32.165 s | 32.156 s | −0.0% |
| I2V | 25.045 s | 24.905 s | −0.6% |
| Policy（生成のみ） | 41.66 s | 42.88 s | +2.9% |
| Policy（conditioning 込み） | **42.44 s**（今回更新） | 42.438 s | — |

conditioning は 81.57 → **0.28 秒**（291 倍）。README §2 と
`docs/cosmos3_rocm_policy_optimization_final_report.md` は更新済み。

### 再現手順の正（DLS-005 / DLS-007 / DLS-009）

経路ごとに必要な条件が異なる。**片方の知識を他方に適用しないこと。**

```bash
# Policy 生成のみ 41.66 秒（cosmos_framework 経路）— TunableOp 表は不要
python scripts/run_cosmos_framework_policy_rocm.py --warmup-runs 2 --policy-condition-cache

# Policy conditioning 込み 42.44 秒
python scripts/run_cosmos_framework_policy_rocm.py --warmup-runs 2 \
  --policy-sync-profile --policy-min-condition-encode

# T2I / T2V / I2V（diffusers 経路）— TunableOp 表が必須
COSMOS3_ROCM_IMAGE=cosmos3-rocm72-diffusers:local \
COSMOS3_DIFFUSERS_INSTALL=true \
python3 scripts/run_rocm_speed_matrix.py --variant aotriton_tuned --case <case> --execute
```

- `aotriton_tuned` = `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` + `PYTORCH_TUNABLEOP_ENABLED=1` +
  `TUNING=0` + `RECORD_UNTUNED=0` + `FILENAME=/workspace/result/rocm_speed_matrix/tunableop_results%d.csv`
- T2I の記録 case は runner に存在しない。手打ちコマンドが
  `docs/cosmos3-rocm-t2i-non-gemm-improvement-plan.md` L289-306 にある
- **落とし穴**: `summary.json` の `tunableop_config.enabled` は CLI 引数のみ反映し環境変数を
  反映しない（`configure_tunableop` L591-597）。記録側も `false` と出るので判断材料にならない

### 測定プロトコルの正（DLS-009、今セッションで新規確定）

- **本環境は同一条件でも run-to-run で bit 再現しない**。生成物の md5 / action 値の一致を
  等価性の判定基準に使ってはいけない。実測フロア（同一フラグ v4 vs v6）は
  action `mean_abs_diff 0.0268` / `max 0.4328`
- 等価性を判定するなら**決定的な中間点**を使う。conditioning 系なら
  `_prepare_inference_data` の戻り値 `initial_noise`（サンプリングへの入力テンソル）。
  同一プロセス内なら通常エンコード 2 回でビット一致するため比較の土台が成立する
- **bit 一致を判定条件に置く前に、対照 run で非決定性フロアを測る**

## 完了済み（今セッション）

- `--policy-min-condition-encode` の E2E 検証（`result/mincond_v5_20260727/`）:
  `generate_batch` 124.906 → 42.438 秒、`get_data_and_condition` 81.575 → 0.276 秒、
  profiler に `min_condition_encode_applied` 記録・`_skipped` 無し
- 対照 run（`result/control_v6_20260727/`、フラグ無し・同一条件）による切り分け:
  v4 と bit 一致せず、その差（mean_abs_diff 0.0268）はフラグ有無の差（0.0139）より大きい
  → 出力差は実装由来ではなく GPU 非決定性
- プローブ v3（`result/conditioning_probe_v3_20260727/probe_v3.json`）で等価性を直接実証:
  `initial_noise` が通常 / 最小エンコードでビット完全一致（max_abs_diff 0.0）、
  一方 `x0_tokens_vision` の latent frame 1〜4 は max_abs_diff 4.08〜4.97 で実際に大きく異なる
  → 当該フレームがサンプリングに到達しないことの直接証拠。DLS-008 の assumption を実測確定に更新
- 前セッション v2 プローブの欠落を発見: docstring は cond_mask 実測を掲げていたが、
  コードは encode 内で `ProbeDone` を送出して到達していなかった（json 出力も無し）
- DLS-009 を起票（判定基準の棄却 / 非決定性フロアの確定 / 等価性の実証）
- 対外文書の更新（ユーザー選択「42.44 秒に更新し根拠を注記」、2026-07-27）:
  README §2 と最終報告の §1 表・IMPORTANT・§3 結論
- todo.md から DLS-007 / DLS-008 で解決済みの残留 2 件を削除

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー・注意事項

- **未 push**: main が `origin/main` より 6 コミット先行
- **未マージ**: `experiment/teacache-quality-eval`。両ブランチとも `.claude/.dls/active.md` を
  変更しているためマージ時に衝突する（main 側 = DLS-009/008/007/006/005/003/002/001、
  experiment 側 = DLS-004/003/002/001）
- **TeaCache 実験の速度数値は報告しない**（DLS-003 制約）。レポートは品質差のみ
- **実測で分岐・条件が確定する前に結論を出さない**（前セッションで同型の誤り 3 回、
  今セッションでも v2 プローブの未実測部分が発覚）。手書きコマンドは `case_command()` の
  生成物と文字列比較し、静的読解の結論はテンソルのダンプで確認してから報告する。
  原本: `.claude/.dls/raw/20260727_chat_repro_protocol_and_conditioning_scope.md`
- **計測値が想定と桁違いなら計測系を疑う**: MIOpen カーネル探索がコールドスタートで
  支配的になる。`MIOPEN_USER_DB_PATH=/workspace/result/rocm_speed_matrix/miopen_user_db` の
  マウントが有効
- Bash 出力が空になる事象が散発。回避策: スクラッチパッドのファイルにリダイレクトして Read

## 関連ファイル

- `.claude/.dls/active.md`（DLS-001〜003, 005〜009。DLS-004 は experiment ブランチ側）
- `.claude/.dls/raw/20260727_doc_mincond_e2e_verification_and_nondeterminism_floor.md`（今回の原本）
- `.claude/.dls/raw/20260727_doc_conditioning_probe_v2_measured.md`（**cond_mask は未実測**。§5 参照）
- `.claude/.dls/raw/20260727_doc_3modes_repro_verification.md`（3 モード再現検証）
- `.claude/.dls/raw/20260727_doc_conditioning_probe_refutes_189frame_claim.md`（189 フレーム説の棄却）
- `.claude/.dls/raw/20260726_doc_conditioning_full_frame_encode_verification.md`（**§2-4 は棄却済み**）
- `scripts/run_cosmos_framework_policy_rocm.py`（`--policy-min-condition-encode`、`_min_frame_encode` /
  `_plan_allows_min_encode` / `_profiled_get_data_and_condition`）
- `result/mincond_v5_20260727/`（採用値 42.44 秒）、`result/control_v6_20260727/`（対照）、
  `result/mainline_full_v4_20260726/`（削減前の基準）、
  `result/conditioning_probe_v3_20260727/probe_v3.json`（等価性の実証データ）
