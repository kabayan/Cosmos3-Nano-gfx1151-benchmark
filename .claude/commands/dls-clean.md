---
name: dls-clean
description: 共有 DLS データの整合性・todo 衛生・archive 候補を承認付きで整理する
---

# /dls-clean — 共有 DLS データ整理

cc / Codex / Antigravity で共通に実行できる DLS 保守入口。配布更新の完了証明は行わない。

## 境界

- 対象: `.claude/.dls/active.md` / `archive.md`、`tasks/todo.md` の候補確認と承認後の整理
- 対象外: `.claude/.dls/raw/`、`.claude/.dls/.migration_state.json`、`update_pending`。読取・変更・解除をしない
- **承認前は一切書き換えない**。候補、根拠、変更対象を提示し、ユーザーの明示承認後だけ変更する

## Phase 1: 整合性確認

`.claude/.dls/skills/02_consistency_check.md` を参照し、必須フィールド、`supersedes` 循環、
`rejected_alternatives` と固定値の整合を確認する。不整合があれば修正候補を提示し、承認後にのみ更新する。

## Phase 2: todo DRY 確認

`tasks/todo.md` がある場合、`.claude/.dls/skills/06_todo_hygiene.md` の **手順 B** を実行する。
重複は行番号・引用・推奨リンクを候補として提示し、承認後にのみリンクへ置換する。

## Phase 3: archive レビュー

`.claude/.dls/skills/05_archive_management.md` の Step 1 / 2.5 を用い、**active.md の全エントリ**を
次のいずれかに分類する。目安件数を超えているときは、未分類のまま完了報告してはならない。

| 分類 | 必要な根拠 | 次の扱い |
|---|---|---|
| 即時archive候補 | 全面的な supersedes | 承認対象として提示 |
| 減衰archive候補 | 古い日付、参照なし、`where` 非活性の3点を確認 | 承認対象として提示 |
| 保持 | 現在のコード・運用・依存関係のいずれかが有効 | 保持理由を記録 |
| 追加証拠収集 | 判定に必要なgit履歴・参照関係が不足 | このPhaseで実行する調査コマンドと判定条件を記録 |

分類は件数をバッチに分けてよいが、各バッチで date / incoming参照 / `where` の更新履歴を確認し、
全件の分類表を出し切るまで次の保守サイクルへ丸投げしない。`active.md` から `archive.md` への移動は、
候補ごとの明示承認後にのみ実行する。

## Phase 4: commit 後の todo 整理（限定再利用）

`dls-commit` から呼ばれた場合のみ、`.claude/.dls/skills/06_todo_hygiene.md` の **手順 A** を実行する。
直前コミットの完了候補と永続化先を提示し、承認後にのみ `tasks/todo.md` から削除する。
この限定手順では Phase 1〜3 を暗黙実行しない。

## 完了報告

候補数、承認済み変更、保留項目を報告する。`update_pending` は変更せず、配布更新が必要なら
cc で `/dls-update` を案内する。
