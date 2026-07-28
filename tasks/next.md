# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28 06:12
> 前セッションの要約: /dls-discuss（v1/v2 精度・速度の乖離と許容範囲）から合否軸を「価格差 2.0 以内」に確定し、guidance 公式デフォルト実測（T2I 4.0 / T2V・I2V 6.0）を実行。公式条件では 3 モードとも 2.0 超過（2.53x/5.25x/11.33x）と確定し、主因の und branch cache CFG スラッシュを特定した（DLS-016）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。**（guidance 実測 3 run は 04:56 UTC 完了、exit 0）

**ブランチは 2 系統のまま**（前回から変化なし）:
- `main`（チェックアウト中）: 未 push コミットが origin/main より先行
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ、active.md 衝突あり

### 合否軸と現在地（DLS-016）

- **合否軸**: 「対記事倍率が価格差 2.0 以内か」（ユーザー決定 2026-07-28。旧「対記事 1.5 倍以内」を置換）
- 現状: **Policy 生成 1.98x のみ内側**（conditioning 込み 2.02x は境界上）。
  T2I/T2V/I2V は公式 guidance 条件で **5.25x / 2.53x / 11.33x と超過**。
  これがチューニングの新ベースライン（「今は収まらない結論でも良い、これをベースに進める」）
- **精度は素の実測提示に凍結**（ユーザー決定）。CUDA 参照 run は当分実施不可の前提に変更

### 機構の確定（DLS-016）

- CFG は逐次 2 回 forward（バッチ倍増ではない）→ TunableOp 表は有効なまま
  （T2V の s/call が 0.767→0.718 で不変が直接証拠）。T2V 2.53x = 純粋な計算量 2 倍
  → CFG 条件のハードウェア素比 ≈2.5 倍
- und branch cache は単一スロット署名式のため CFG 下で **140 calls / 140 writes / 0 reads**
  の全スラッシュ（T2I s/call 4.5 倍・I2V 9.6 倍悪化。
  `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py` L871-900）
- 2 スロット化で回復しても概算 T2I ≈2.4x / I2V ≈2.6x で **まだ 2.0 超**。
  per-call 約 20% の追加短縮が本丸

### 記事 per-step MSE チャートの発見（README 更新時に併記予定）

- 記事側（DGX）も **step 6-7 で per-step MSE ≈0.10 と公式しきい値 0.05 を局所超過**。
  全体 0.0132 は「2 step 以外ほぼゼロ」構造での合格。画像アップロード 2026-05-22（squash 窓内側）
- 本環境 v2 の最大逸脱も同じ step 6-7（0.48/0.45、約 4.5 倍）+ 広帯域誤差。
  per-step 表は `raw/20260728_doc_guidance_official_measurement.md` §6

### 実測資産

- `result/guidance_official_20260728/`: run_commands.sh + summary 3 点は **force-add 済み**、
  出力 jpg/mp4 は untracked。guidance 1.0 側の対照は `result/verify_3modes_v3_20260726/`

## 完了済み（今セッション）

- /dls-discuss: v1 速度は乖離なし（E4 定常 1.15 s/it = 本線帯 1.128〜1.175、E4 総時間はプロトコル
  不一致の無効値）、v1 精度乖離は E4 で確定済み、v2 許容性は「素の提示」で凍結
  — 議論ノート `raw/20260728_chat_v1v2_accuracy_and_price_criterion.md`
- guidance 公式デフォルト実測 3 run（DLS-007 v3 プロトコル、TunableOp 表 md5 確認済み、全 exit 0）
  → DLS-010 dormant 解消 — 実測原本 `raw/20260728_doc_guidance_official_measurement.md`
- 記事 per-step MSE チャート発見 + 本環境 per-step 計算（`scripts/check_policy_golden_mse.py` の golden 使用）
- DLS-016 起票、todo.md 更新（guidance 実測項目を完了削除、README 項目更新、/dls-plan 項目追加）

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）
（先頭候補: CFG 条件チューニングの仮説選定 `/dls-plan`、README §2 両条件併記への更新）

## ブロッカー・注意事項

- CUDA 参照 run は当分実施不可（ユーザー決定 2026-07-28）。golden 帰属の決着はペンディングのまま
- 記事の実際の guidance は依然未確認（DLS-010 assumption、confidence medium のまま）。
  ただし 1.0 / 公式値の両条件が実測済みで、どちらに倒れても対応値がある
- negative prompt はスクリプト既存値で公式 `neg_prompts.json` と別物（uncond 系列長が短い）。
  CFG 条件の再測定・品質比較時に扱いを決める
- und cache 2 スロット化を実装する場合、対象はイメージ同梱 `/opt/diffusers` と
  `third_party/diffusers` の両方（乖離させない）

## 関連ファイル

- `.claude/.dls/active.md`（DLS-016）
- `.claude/.dls/raw/20260728_doc_guidance_official_measurement.md`（実測原本）
- `.claude/.dls/raw/20260728_chat_v1v2_accuracy_and_price_criterion.md`（議論原本）
- `result/guidance_official_20260728/`（summary・出力・再現スクリプト）
- `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py`（und cache）
- `third_party/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`（CFG 実装）
- `README.md` §2（更新待ち）
