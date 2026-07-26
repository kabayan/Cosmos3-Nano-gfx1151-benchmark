---
name: dls-discuss
description: 議論モード起動。引数 [topic] で過去議論を検索し継続/新規開始。応答態度を切替（反対視点必須・「何もしない」候補必須・追従バイアス自己検出）。終了時の保存は /dls-commit Phase 1.6 (skill 07) が担う。
---

# /dls-discuss [topic] — 議論モード

「DLS で仮説の取り扱いを明示する」議論を、応答態度を切替えた状態で進める。
終了時の raw/discussion 保存と DLS 化は `/dls-commit` Phase 1.6 が担うため、
このコマンドは **議論モードへの入り口** として動作する。

---

## Step 1: topic の解決

- 引数あり: `/dls-discuss "追従バイアス対策"` → topic 確定
- 引数なし: 会話冒頭から CC が topic を推測し、ユーザーに確認

---

## Step 2: 過去議論の検索（議論済み判定）

以下を並列で実行:

- `grep -l "<topic キーワード>" .claude/.dls/active.md .claude/.dls/archive.md`
- `ccc search "<topic>"`（cocoindex-code、ある場合）
- `ls .claude/.dls/raw/ | grep -i "<topic>"`（discussion / chat 種別の過去ノート）

検出結果に応じて分岐:

### 該当あり（DLS / raw が見つかった）

```
過去に議論済みの可能性があります:
- DLS-XXX: <what 抜粋>
- raw/<file>: <ファイル名>

どう扱いますか?
  c) 継続: 過去 DLS を context に読み、議論を再開
  n) 新規: 過去は参照せず、別 topic として開始
  r) raw も参照: DLS まとめ + raw 全文を context に読む
```

- **c (継続)**: 関連 DLS エントリを Read（active.md の該当 section）。raw は読まない（cold に保持）
- **n (新規)**: 過去 DLS / raw は参照しない
- **r (raw 参照)**: DLS + raw/discussion ファイルを Read

### 該当なし

新規議論として開始。Step 3 へ。

---

## Step 3: 議論モードの応答態度（議論中ずっと適用）

ユーザーが `/dls-commit` で解除するまで、以下の規律で応答する:

- **反対視点を必ず 1 つ併置する**（同意のみで終わらせない）
- **「何もしない」候補を必ず含める**（YAGNI 適用）
- **賛成 / 反対 / 留保を明示する**（整理だけで終わらない）
- **反対できない時は「反対できない、なぜなら〜」と明言**（形だけの反対は禁止）
- **「全面同意」を出す前に追従バイアスを自己検出する**

これらは `claude-md-fragment.md` の「モード別の応答態度」と整合する。

---

## Step 4: 解除（/dls-commit に委譲）

このコマンドは解除フローを持たない。`/dls-commit` を呼ぶと:

- Phase 1.6 (skill 07) が議論ノートの抽出を判定
- 抽出する場合は `.claude/.dls/raw/YYYYMMDD_chat_<topic>.md` に保存
- 関連 DLS エントリの作成 / 更新 / supersedes をユーザー確認のうえ実行
- `rejected_hypothesis` を必要に応じて記載（dls-entry.md ルール）

専用の `/exit-discuss` は不要。`/dls-commit = フェーズ完了` の意味付けに統一。

---

## 注意

- 議論中に新しい仮説を選定する段階に進む場合は `/dls-plan` に切替を提案する
- 実装系の指示（「実装して」「修正」等）が来たら議論モードを自動解除し執行集中に戻る
- raw 読み込みは明示指示時のみ（DLS 哲学: エントリはキャッシュ、原本は再生成不能）
