# 次のセッションへの引き継ぎ

> 作成日時: YYYY-MM-DD HH:MM
> 前セッションの要約: （1〜3 文で）

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。本ファイルの「次のアクション」セクションは todo.md への
参照リンクのみとし、タスク文言を直接書かない。

---

## 現在の状態

- ブランチ: `xxx`、`origin/xxx` から N コミット先行 / 遅れ
- 未コミット変更: なし / あり（…）
- 直近コミット: `<sha>` <subject>

## 完了済み（今セッション）

- （文脈として残したい補足のみ。コミット履歴で十分なものは省略）

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー / 注意事項

- （次セッションで詰まりそうな点。なければ「なし」）

## 関連ファイル

- （作業中のファイルパス。コミット差分で十分なら省略）

## DLS 議論モード状態

`/dls-discuss` または `/dls-plan` で議論モードに入っている場合のみ記入。
`/dls-commit` 完了時に削除する。

```yaml
mode: discuss        # discuss / plan / なければセクション削除
topic: <議論の主題>
raw_path: .claude/.dls/raw/YYYYMMDD_chat_<topic>.md   # 既存の場合
started_at: YYYY-MM-DDTHH:MM
last_active: YYYY-MM-DDTHH:MM
```

## 未確定 DLS 草案

`/dls-discuss` `/dls-plan` で合意したが active.md に未追加の草案。
`/dls-commit` Phase 1.6 で確定 / 削除する。

### 草案 1: <タイトル>

- proposed_id: DLS-XXX（次の連番候補）
- what: <仕様>
- why:
  - origin: user_request / user_confirmed / implementation
  - business: <動機>
- where: <影響範囲>
- depends_on: <DLS-XXX>
- supersedes: <DLS-XXX>（あれば）
- rejected_hypothesis:（supersedes がある場合は併記）
  - target: DLS-XXX
  - hypothesis: <棄却仮説 1 行>
  - reason: <根拠>
- rejected_alternatives:
  - <案A>: <dormant 理由>
- assumption: <未検証前提> (confidence: high/medium/low)
