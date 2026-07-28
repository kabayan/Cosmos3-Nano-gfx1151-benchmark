# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28 09:45
> 前セッションの要約: und branch cache 2 スロット化を実装・検証し DLS-018 で確定（出力ビット一致のまま T2I 5.25x→2.25x / I2V 11.33x→2.69x）、続けて README §2 を価格差 2.0 基準 + guidance 両条件併記 + 精度の素の提示に更新した（`5c847ff`）。/dls-discuss「チューニング履歴再分析」は候補 A〜E 提示のまま未決着。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** 作業ツリーはクリーン（`5c847ff` + 本コミット）。

**ブランチは 2 系統のまま**:
- `main`（チェックアウト中）: 未 push コミットが origin/main より先行
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ、active.md 衝突あり（**DLS-004 が main の active.md に不在**の帳簿不整合。議論ノート候補 B の当事者）

### 対外文書の現状（README 更新完了後）

- README §2: 合否基準は**価格差 2.0 以内**。guidance 1.0（1.23x/1.46x/1.47x）と公式 guidance
  （**T2I 2.25x / T2V 2.53x / I2V 2.69x**、2 スロット化後）の両条件併記済み。Policy 1.98x は基準内側
  （conditioning 込み 2.02x は境界上）、公式 guidance の 3 モードは超過（CFG 計算量 2 倍が要因、
  追加短縮余地なしは GEMM 実測済み = DLS-017）
- README §2 NOTE: 精度は素の提示 — golden MSE 0.126〜0.134 vs 基準 0.05 / 記事 0.0132、
  帰属未決着（CUDA 参照 run 当分不可）、記事側も step 6-7 per-step 0.05 超過、を明記済み
- README §4: ピーク TFLOPS の形状依存注記（実運用 GEMM 最大 36.11 TF）追加済み
- docs/cosmos3_rocm_optimization_analysis.md: 「論文値」→「記事値」修正済み
- 「対記事 1.5 倍以内」表記は README から除去済み。docs/ の旧最終報告
  （cosmos3_rocm_policy_optimization_final_report.md）は歴史的文書のため未変更

### DLS-018（2 スロット化）の要点

- 実装: `third_party/diffusers/.../transformer_cosmos3.py`（クローン内コミット `f829105c7`）を
  署名キー LRU 2 スロットに変更。検証: 2 writes / 138 reads、T2I jpg / I2V mp4 md5 ビット一致、
  記録 `result/guidance_2slot_20260728/`
- **イメージ同期済み**: `cosmos3-rocm72-diffusers:local` = `sha256:554e0573ec89...`
  （docker cp + commit 方式。旧 `sha256:eab19ad6eb66...` はロールバック用。
  third_party を変更したら同方式で再同期、依存パッケージ変更時のみ rebuild）
- T2V への cache 適用（DLS-017 C 案）は dormant のまま

### 議論の中断状態（/dls-discuss → 未決着）

- topic: 過去のチューニング履歴を公平に再分析し改善点がないか検討
- 原本: `raw/20260728_chat_tuning_history_reanalysis.md`（失敗 3 類型・候補 A〜E・CC 賛否）
- 候補 A（claim 残件の吸収）は README 更新で**実質完了**。残る判断は B（DLS-004 ブランチ整合 +
  CK FMHA 追跡方法）/ D（lessons.md 新設 or 参照削除）/ E（何もしない）
- DLS 未起票（採用判断が無いため）。再開: `/dls-discuss チューニング履歴再分析`

## 完了済み（今セッション）

- und branch cache 2 スロット化: 実装 + イメージ同期 + 検証（3 基準合格）— **DLS-018**、
  原本 `raw/20260728_doc_und_cache_two_slot_verification.md`、コミット `f188400`
- /dls-discuss（チューニング履歴再分析）→ 候補 A〜E 提示、議論ノート保存（`fb5ed51`）、未決着
- README §2/§4 + docs 表記の対外表現更新 — コミット `5c847ff`（新規 DLS なし、既存判断の執行）
- todo.md 整理: 2 スロット化タスク・README 更新タスクを承認のうえ削除

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）
（先頭: 議論候補の選択（ユーザー判断）。候補 A は README 更新で実質完了済みのため B/D/E が残り）

## ブロッカー・注意事項

- CUDA 参照 run は当分実施不可（ユーザー決定 2026-07-28）。golden 帰属の決着はペンディング
- 記事の実際の guidance は依然未確認（DLS-010 assumption、confidence medium）。README は両条件併記で対応済み
- **third_party/diffusers とイメージ /opt/diffusers を乖離させない**（docker cp + commit で同期）
- この環境では Bash の grep / git 出力が時折無出力・整形される事象あり（python subprocess で代替）
- 2.0 到達を目的化して計算省略系（TeaCache 等）に手を出すのは DLS-003 でユーザー棄却済み
- 次の Policy run 時にカーネルキャッシュ持ち越し（DLS-015）の golden MSE 帯確認を便乗実施（todo 参照）

## 関連ファイル

- `README.md` §2 / §4（更新済み）、`docs/cosmos3_rocm_optimization_analysis.md`（更新済み）
- `.claude/.dls/active.md`（DLS-016 / DLS-017 / DLS-018）
- `.claude/.dls/raw/20260728_doc_und_cache_two_slot_verification.md`（実装・検証原本）
- `.claude/.dls/raw/20260728_chat_tuning_history_reanalysis.md`（議論原本、候補 B/D/E が未決着）
- `result/guidance_2slot_20260728/`（2 スロット化後の実測記録）
- `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py`（クローン内 `f829105c7`）
