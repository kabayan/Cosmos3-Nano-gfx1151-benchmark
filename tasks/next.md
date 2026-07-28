# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28 06:46
> 前セッションの要約: /dls-discuss で合否軸を「価格差 2.0 以内」に確定し guidance 公式デフォルトを実測（DLS-016）、続く /dls-plan で CFG バッチ化仮説を GEMM PoC により棄却し、採用案を und branch cache の 2 スロット化に確定した（DLS-017）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** 作業ツリーはクリーン（`9ebf851`）。

**ブランチは 2 系統のまま**（前回から変化なし）:
- `main`（チェックアウト中）: 未 push コミットが origin/main より先行
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ、active.md 衝突あり

### 合否軸と到達可能性（DLS-016 → DLS-017 で更新）

- **合否軸**: 「対記事倍率が価格差 2.0 以内か」（ユーザー決定 2026-07-28）
- 公式 guidance 条件の実測: T2V **2.53x** / T2I **5.25x** / I2V **11.33x**（3 モードとも超過）。
  Policy 生成 1.98x のみ内側（conditioning 込み 2.02x は境界上）
- **DLS-017 で「CFG 条件での 2.0 到達は同一計算内容の制約（DLS-003）下では非現実的」と確定**。
  チューニングの目的は「2.0 到達」ではなく **スラッシュ由来の異常悪化の解消** に変更済み
- **精度は素の実測提示に凍結**（ユーザー決定）。CUDA 参照 run は当分実施不可

### DLS-017 PoC の要点（B 棄却の根拠）

- GEMM マイクロベンチ（`scripts/probe_cfg_batching_gemm.py`、実形状 4 種 × トークン数 5 種、
  TunableOp 無効で土台統一）で **トークン 2 倍の所要比 ratio = 全体 1.926 / 実運用帯 N≥672 で 1.962**
- 達成 TFLOPS が N と 2N でほぼ不変・GB/s は単調低下 = **演算律速の署名**。
  バッチ化の利得は約 1.9% で必要な 20% に遠く届かない
- cond/uncond は系列長が異なるためバッチ化しても総トークン数は不変。
  「2 回呼ぶこと」ではなく「計算量が 2 倍になること」が本質だった
- 副次: 観測 GEMM 最大 **36.11 TFLOPS** > README §4 bench_peak **20.91 TFLOPS**（16384³）。
  GEMM 経路に 20% の余地なし。README のピーク値を上限根拠に使う際は形状依存の注記が要る

### 次の実装対象（採用案 A）の前提

- 現状: CFG 下で und branch cache が **140 calls / 140 writes / 0 reads** の全スラッシュ
- 実装先: `third_party/diffusers/.../transformer_cosmos3.py`（`_und_branch_cache` L496-540 の
  単一スロット構造、L871-900 の署名判定）。**イメージ同梱 `/opt/diffusers` と同期必須**
- 概算回復 T2I 5.25x→≈2.4x / I2V 11.33x→≈2.6x は DLS-017 assumption（confidence: medium）。
  反証手段は実装後の実測 1 run

## 完了済み（今セッション）

- /dls-discuss（v1/v2 精度・速度の乖離と許容範囲）→ 合否軸を価格差 2.0 に確定、精度は素の提示に凍結
  — 議論原本 `raw/20260728_chat_v1v2_accuracy_and_price_criterion.md`
- guidance 公式デフォルト実測 3 run（DLS-007 v3 プロトコル、全 exit 0）→ DLS-010 dormant 解消
  — 実測原本 `raw/20260728_doc_guidance_official_measurement.md`、**DLS-016** 起票
- 記事 per-step MSE チャート発見（記事側も step 6-7 で per-step 0.05 超過、全体 0.0132）
  + 本環境 v2/v1 の per-step 計算
- /dls-plan（CFG 条件チューニング）→ GEMM PoC で候補 B 棄却、A を採用
  — PoC 原本 `raw/20260728_doc_cfg_batching_gemm_poc.md`、**DLS-017** 起票
- コミット 2 本: `6c94fd6`（DLS-016）、`9ebf851`（DLS-017）

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）
（先頭候補: und branch cache 2 スロット化の実装、README §2 両条件併記への更新）

## ブロッカー・注意事項

- CUDA 参照 run は当分実施不可（ユーザー決定 2026-07-28）。golden 帰属の決着はペンディング
- 記事の実際の guidance は依然未確認（DLS-010 assumption、confidence medium）。
  ただし 1.0 / 公式値の両条件が実測済みで、どちらに倒れても対応値がある
- negative prompt はスクリプト既存値で公式 `neg_prompts.json` と別物（uncond 系列長が短い）
- A 実装時は `third_party/diffusers` と イメージ同梱 `/opt/diffusers` を乖離させない
- 2.0 到達を目的化して計算省略系（TeaCache 等）に手を出すのは DLS-003 でユーザー棄却済み。
  やるなら仕様レベルの supersede が必要（勝手に再提案しない）

## 関連ファイル

- `.claude/.dls/active.md`（DLS-016 / DLS-017）
- `.claude/.dls/raw/20260728_doc_cfg_batching_gemm_poc.md`（PoC 原本）
- `.claude/.dls/raw/20260728_doc_guidance_official_measurement.md`（guidance 実測原本）
- `.claude/.dls/raw/20260728_chat_v1v2_accuracy_and_price_criterion.md`（議論原本）
- `scripts/probe_cfg_batching_gemm.py` / `result/cfg_batch_probe/gemm_bf16.json`
- `result/guidance_official_20260728/`（summary・出力・再現スクリプト）
- `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py`（und cache = A の実装先）
- `third_party/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`（CFG 実装）
- `README.md` §2（両条件併記の更新待ち）/ §4（ピーク TFLOPS の注記候補）
