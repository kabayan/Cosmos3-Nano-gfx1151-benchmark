---
name: dls-codex-review
description: codex CLI の異モデルコードレビューを実行する薄ラッパー。デフォルトで未コミット変更を対象 (--uncommitted)。--base/--commit/--dls 指定で他形式に切替。Opus 4.7 の手抜き検知 + コミット前自動レビュー (DLS-154)。
---

# /dls-codex-review [--base BRANCH | --commit SHA] [--dls DLS-XXX,DLS-YYY,...] [追加 PROMPT]

codex CLI の `codex review` サブコマンドを起動して異モデルレビューを得る薄ラッパー。
レビュー結果は stdout に出力し、CC が読んで重要指摘を tasks/todo.md 追記提案する。

`/review` (PR 用) / `/ultrareview` (cloud / billed) / `/security-review` (security) と並列の
コードレビュー系 skill。codex CLI を呼ぶことを命名で明示する（DLS-154 結論）。

## 前提

- codex CLI ≥ 0.128.0 がインストール済み（`codex review --uncommitted/--base/--commit` 提供 version）
- 不在時は `docs/setup.md#codex` への誘導を出してエラー終了

## 引数

| 引数 | 動作 |
|---|---|
| 引数なし | `codex review --uncommitted` (デフォルト、コミット前レビュー動機に直結) |
| `--base <BRANCH>` | `codex review --base <BRANCH>` にフォワード (ブランチ全体レビュー) |
| `--commit <SHA>` | `codex review --commit <SHA>` にフォワード (過去コミットレビュー) |
| `--dls DLS-XXX,DLS-YYY,...` | active.md から該当エントリを抽出して PROMPT として codex に渡す |
| 残余 PROMPT 文字列 | codex の `[PROMPT]` 位置引数として渡す (カスタムレビュー指示) |

`--uncommitted` / `--base` / `--commit` は排他。同時指定は codex 側でエラー。

## Step 1: codex 不在確認

```bash
if ! command -v codex >/dev/null 2>&1; then
  echo "エラー: codex CLI が PATH にありません。"
  echo "  → docs/setup.md#codex を参照してインストールしてください"
  echo "  → 動作確認 version: codex-cli 0.128.0 以上 (DLS-132 PoC #4)"
  exit 1
fi
```

## Step 2: 引数パース

引数を解析して以下の変数を組み立てる:

- `MODE` = `--uncommitted` / `--base <BRANCH>` / `--commit <SHA>` のいずれか (デフォルト `--uncommitted`)
- `DLS_IDS` = カンマ区切りの DLS-XXX 列 (空の場合は context 注入なし)
- `EXTRA_PROMPT` = 残余文字列 (空可)

## Step 3: --dls 指定時の context 抽出

`--dls DLS-XXX,DLS-YYY,...` 指定時、active.md から該当エントリを切り出して PROMPT として codex に渡す:

```bash
DLS_CONTEXT=""
for id in $(echo "$DLS_IDS" | tr ',' ' '); do
  # active.md エントリ見出しの規範は h2 単独行 `## DLS-XXX` (templates/entry.md / L008)
  section=$(awk "/^## ${id}\$/,/^## DLS-[0-9]+\$/" .claude/.dls/active.md | sed '$d')
  DLS_CONTEXT="${DLS_CONTEXT}${section}\n\n"
done
```

抽出した DLS_CONTEXT を `EXTRA_PROMPT` の前に prepend する:

```
以下の判断ログ (DLS エントリ) を踏まえてレビューしてください:

<DLS_CONTEXT>

追加指示: <EXTRA_PROMPT>
```

CLAUDE.md 全文や active.md 全体は渡さない (codex トークン過多回避、DLS-154 で「フル context」棄却済)。

## Step 4: codex 起動

```bash
if [ -n "$DLS_CONTEXT" ] || [ -n "$EXTRA_PROMPT" ]; then
  # PROMPT 指定あり: stdin から読ませる
  echo -e "$FULL_PROMPT" | codex review $MODE -
else
  # PROMPT なし: 素のレビュー
  codex review $MODE
fi
```

stdout に codex のレビュー verdict + 指摘内容が出力される。

## Step 5: CC による指摘抽出と todo.md 追記提案

CC は codex の stdout を読み、重要指摘を抽出する。

- **何が重要か**: バグの可能性 / セキュリティ / DLS 棄却済み案の再導入 / 既存テストとの矛盾
- **何が重要でないか**: 命名好みのリファクタ提案 / 形式論的な YAGNI 違反指摘 (CLAUDE.md §1 と整合する場合のみ採用)

抽出した指摘を「tasks/todo.md の Active に追記提案しますか？」とユーザーに確認:

- Y → todo.md に `[ ] codex 指摘: <要約> (出典: codex review YYYY-MM-DD)` の形で追記
- N → 表示のみで終了

## DLS との連携

- **`/dls-commit` 統合はしない** (独立 skill、DLS-154 結論)。レビューしたい時だけ呼ぶ。
- **`--dls` 手動指定 5 回利用で 0 件なら where 自動抽出検討** (DLS-154 assumption (5) falsify trigger)
- **指摘が常に空 or 些末なら DLS-154 動機自体を再検討** (DLS-154 assumption (2)(3) falsify trigger)

## 外部依存 (DLS-151)

- claude-code 規約: frontmatter `name` / `description`
- codex CLI ≥ 0.128.0 (`codex review` サブコマンド、`--uncommitted` / `--base` / `--commit` オプション)
- active.md パス: `.claude/.dls/active.md` (claude-code 配下、codex / gemini 環境では別 path 規約となる可能性あり、§24 Phase 3 で adapter 検討)

## 関連 DLS

- DLS-154: 本 skill の方針確定 (B 案 = codex review 薄ラッパー採用)
- DLS-155: 実装計画の仕様詰め (init.py 拡張 / dogfood / version bump)
- DLS-132: codex-cli 0.128.0 hook 動作確認 (本 skill が利用する version の検証実績)
- DLS-076: §24 Phase 3 codex / gemini 対応 CLI 抽象化 (本 skill が先行 dogfood)
- DLS-151: CLI ハーネス依存明示ルール
