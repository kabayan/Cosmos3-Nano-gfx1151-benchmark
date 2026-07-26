---
name: dls-continue
description: tasks/next.md を読み込み、前セッションの作業を再開する
---

# 作業再開

前セッションの引き継ぎ情報から作業を再開してください。

## 手順

1. `tasks/next.md` を読み込む（存在しない場合はユーザーに報告して終了）
2. `tasks/todo.md` を読み込む（存在する場合）
3. `.claude/.dls/active.md` を読み込む（既にSessionStartフックで注入済みの場合はスキップ可）
4. `tasks/lessons.md` を読み込む（存在する場合）
5. **next.md ↔ todo.md 整合チェック（DLS-123）**: `.claude/.dls/skills/06_todo_hygiene.md` の **手順 C** を実行する
   - next.md の「次のアクション」が「→ `tasks/todo.md` の Active 参照」のみなら何もしない
   - 旧形式（タスク文言が直接書かれている）が残っていれば todo.md の Active に移送 → next.md を参照リンクに置換

## 再開時の確認

上記の情報を読み込んだ後、以下を簡潔に報告してください:

### 状態サマリー
- 前セッションの要約（next.md から）
- 次のアクション（todo.md の Active から、優先順に）
- ブロッカーの有無

### 即座に着手

報告後、**todo.md の Active 最上位タスク**に着手してください。
ユーザーに「何をしますか？」と聞かず、自律的に作業を開始してください。

## ルール

- next.md が古い（2日以上前）場合はその旨を警告し、最新の状態を `git log` / `git status` で確認する
- next.md のブロッカーが解消されているか確認してから着手する
- CLAUDE.md のワークフロールールに従う（3ステップ以上ならプランモード等）
- タスク本体は **todo.md の Active が真のソース**。next.md は文脈・状態の運搬に専念（DLS-123）
