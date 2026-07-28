# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28 12:28
> 前セッションの要約: /dls-discuss「チューニング履歴再分析」を継続し決着（B + A' 採用 = DLS-019、`284fa1b`）。DLS-004 を main の active.md に復元、DLS-017 の欠落見出しを補修、再評価レポート §1 に基準値訂正 CAUTION 注記を追加した。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** 作業ツリーはクリーン（`284fa1b` + 本コミット）。
議論モードは解除済み（チューニング履歴再分析は決着、未決着の議論なし）。

**ブランチは 2 系統のまま**:
- `main`（チェックアウト中）: 未 push コミットが origin/main より先行
- `experiment/teacache-quality-eval`（`eed9aa0` 起点、実装 `3d3b514`）: 未マージ。
  **DLS-004 帳簿不整合は解消済み**（main の active.md に復元、本文は両側同一のため
  将来の merge 衝突は自明に解消できる = DLS-019 assumption）

### チューニング履歴再分析の決着内容（DLS-019）

- 採用 B: DLS-004 復元 + DLS-017 見出し補修（active.md の参照切れ 2 件解消）
- 採用 A': `docs/cosmos3_rocm_further_speedup_reassessment_20260726.md` §1 に CAUTION 注記
  （「対論文比 1.44 倍達成済み」「論文値 8.0 秒」は一次出典なき基準 29 秒由来、DLS-006 是正 +
  DLS-016 合否軸変更を明記。表本体はスナップショットとして原文維持）
- 棄却 C: stale check 再実施（7/26 完了済み・情報増分ゼロ）
- 留保 D: lessons.md の決着（todo.md Active に残置、ユーザー設計判断待ち）
- claim 残件のうち README §4 TFLOPS 注記・optimization_analysis.md「論文値」は処理済みと判明
  （実処理は再評価レポート 1 件のみだった）
- 原本: `raw/20260728_chat_tuning_history_reanalysis.md`（決着追記済み）

### 対外文書の現状

- README §2: 価格差 2.0 基準 + guidance 両条件併記 + 精度の素の提示（`5c847ff`、変更なし）
- README §4: ピーク TFLOPS 形状依存注記あり
- docs の旧最終報告（final_report）と再評価レポートは、いずれも冒頭/該当節に訂正 CAUTION
  注記つきの歴史的スナップショットとして維持

## 完了済み（今セッション）

- /dls-discuss 継続 → 候補 A〜E の前提再検証（A の吸収先失効・DLS-017 見出し欠落の新発見・
  D の前提半減を確認）→ ユーザー選択 B + A' → 執行 → DLS-019 起票 — コミット `284fa1b`
- todo.md 衛生: 決着済み項目を承認のうえ削除、留保項目（lessons.md / CK FMHA 追跡）を独立行で残置

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）
（先頭は留保継続の lessons.md 決着。実行系では TeaCache 品質評価の残り 7 run が最大の未完タスク）

## ブロッカー・注意事項

- CUDA 参照 run は当分実施不可（ユーザー決定 2026-07-28）。golden 帰属の決着はペンディング
- 記事の実際の guidance は依然未確認（DLS-010 assumption、confidence medium）。README は両条件併記で対応済み
- **third_party/diffusers とイメージ /opt/diffusers を乖離させない**（docker cp + commit で同期）
- この環境では Bash の grep / git 出力が時折無出力・整形される事象あり（今セッションも再現。
  無出力が続く場合は python subprocess で代替）
- 2.0 到達を目的化して計算省略系（TeaCache 等）に手を出すのは DLS-003 でユーザー棄却済み
- 次の Policy run 時にカーネルキャッシュ持ち越し（DLS-015）の golden MSE 帯確認を便乗実施（todo 参照）

## 関連ファイル

- `.claude/.dls/active.md`（DLS-019 起票、DLS-004 復元、DLS-017 見出し補修）
- `.claude/.dls/raw/20260728_chat_tuning_history_reanalysis.md`（議論原本、決着追記済み）
- `docs/cosmos3_rocm_further_speedup_reassessment_20260726.md` §1（CAUTION 注記追加）
- `tasks/todo.md`（Active 先頭に lessons.md 留保項目）
