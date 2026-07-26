# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-26 16:53
> 前セッションの要約: /dls-plan で「さらなる高速化余地の再調査」スコープを確定（DLS-001）、
> 5 観点の並列 Web 調査 + ローカル確認を実施して docs/ にレポート化。vLLM 統合仮説を棄却
> （DLS-002）、計算省略系は本線から除外し TeaCache 品質差評価のみ実施と決定（DLS-003）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

- ブランチ: `main`、`origin/main` から 2 コミット先行（未 push）
- 未コミット変更: なし
- 直近コミット: `2719212` Record impl commit SHA in DLS-001..003 entries /
  `51c1b4e` Add further speedup reassessment report and DLS decision log (DLS-001..003)
- DLS: active.md に DLS-001〜003（全て 2026-07-26 起票）。archive.md は未作成（0 件）

## 完了済み（今セッション）

- cocoindex-code (`ccc`) を本プロジェクトで初期化・インデックス構築済み（843 files / 14,802 chunks。`ccc search "<クエリ>"` が利用可能）
- 調査レポート `docs/cosmos3_rocm_further_speedup_reassessment_20260726.md` 作成
  - 結論: 1.127s/step の物理限界は 2026-07 時点でも実質有効。確実な伸び代は
    torch 2.13.0 更新 + クロック張り付き確認の数%規模のみ
  - 重要事実: Cosmos3-Nano は **head_dim=128**（AOTriton PR #200 の gfx1151 tuning DB
    は hdim=64 限定で非適用）。vllm-cosmos3 は Reasoner 用で Policy に転用不可
- DLS-001（調査スコープ）/ DLS-002（vLLM 棄却）/ DLS-003（計算省略系は本線不可、
  TeaCache は品質差評価のみ）を起票

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー / 注意事項

- **DLS-003 の制約を厳守**: TeaCache / INT8 attention は速度成果として本線（README /
  対外比較）に載せない。品質差（アクションテンソル誤差・PSNR/LPIPS）の測定のみ
- TeaCache 品質評価は本線ベンチ環境を汚さないこと（別ブランチ or フラグ分離、
  DLS-003 assumption 参照）
- origin へ 2 コミット未 push（push はユーザー判断待ち）
- Bash 出力が空になる/圧縮される事象が本セッションで散発（原因未特定）。回避策:
  出力をスクラッチパッドのファイルにリダイレクトして Read する

## 関連ファイル

- docs/cosmos3_rocm_further_speedup_reassessment_20260726.md（調査レポート本体）
- .claude/.dls/active.md（DLS-001〜003）
- .claude/.dls/raw/20260726_discussion_further_speedup_scope.md（スコープ議論の原本）
- .claude/.dls/raw/20260726_chat_teacache_scope_decision.md（DLS-003 のユーザー指示原本）
- scripts/run_cosmos_framework_policy_rocm.py（TeaCache 品質評価で触る推論経路。
  attention は :70 の `_flash_attention_forward`、compile は :262-299）
