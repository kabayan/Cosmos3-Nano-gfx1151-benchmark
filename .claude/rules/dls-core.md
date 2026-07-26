# DLS: コア運用リファレンス

DLS（Discovery Linked Specification）システム全体の運用ガイド。

## 参照タイミング

- DLS 全体像や配布アーキテクチャを確認したいとき
- どの skill を呼ぶべきか迷ったとき
- `/dls-update` / `/dls-recover` / `init.py` を実行する前
- 月次メンテナンス時

日常のコード編集時は `rules/dls-code.md`、エントリ記述時は `rules/dls-entry.md` が担当する。本ファイルは全体像ガイドで、それらと役割が重複しない範囲に限定する。

## 3層構造

DLS は以下の 3 層に分離される。

| 層 | 配置 | 役割 | 変更可否 |
|---|---|---|---|
| L1 配布コア | `dls-core/`（dls 配下） | `manifest.json` と配布物（hooks / commands / rules / skills / templates / claude-md-fragment.md） | dls 本体更新でのみ変更 |
| L2 プロジェクト適用 | 各プロジェクトの `.claude/` | `init.py` により L1 から配布される運用ファイル。CLAUDE.md のマーカー内は L1 と同期 | ユーザー編集可（マーカー外のみ） |
| L3 稼働データ | `.claude/.dls/active.md`, `archive.md`, `raw/`, `.migration_state.json` | 判断ログ・原本・migration 適用状態 | 追記のみ（上書き禁止） |

`protected_paths`（`manifest.json` 参照）は init.py が絶対に触らない。L2 ユーザー設定（例: `settings.local.json` の通知 URL）と L3 全体が該当する。

## ディレクトリの役割

- `.claude/.dls/active.md` — 現在有効な判断エントリ（目安 30 件。超過時は減衰候補レビュー、Phase 終了 or コミット集約イベントで `skills/05_archive_management.md` を実行 / DLS-144 軟上限化）
- `.claude/.dls/archive.md` — superseded / 減衰したエントリの退避先
- `.claude/.dls/raw/` — 原本（ヒアリングメモ・メール等）。追記のみ、編集・削除禁止
- `.claude/.dls/skills/` — 運用手順書（01〜05）
- `.claude/.dls/templates/entry.md` — 新規エントリのフルテンプレート
- `.claude/.dls/.migration_state.json` — 最後に適用した migration ID を保持

## Skill 呼び分けマップ

| 状況 | skill |
|---|---|
| ヒアリング / メール / チャットから新規エントリを作成 | `01_generate_entry.md` |
| 月次メンテナンス（必須フィールド・減衰検出） | `02_consistency_check.md` |
| 変更リクエスト受領時の影響範囲分析 | `03_impact_analysis.md` |
| active.md から仕様書を再生成 | `04_spec_generation.md` |
| active.md が目安 30 件超過 → Phase 終了 or コミット集約イベントで減衰再評価 | `05_archive_management.md` |

## 更新・復旧コマンド

- `init.py` — 新規プロジェクトの初回セットアップ、および `manifest.json` 変更の適用
- `/dls-update` — 既存プロジェクトに最新 dls-core を反映し、CLAUDE.md マーカー同期・migration・完了証明を行う
- `/dls-clean` — 共有 DLS データの整合性・todo 衛生・archive を候補提示と承認付きで整理する
- `/dls-recover` — 欠損ファイルの選択復元（protected_paths の L3 は対象外）

配布更新は `/dls-update`、日常の共有データ保守は `/dls-clean` を使う。`/dls-recover` は hooks 削除・settings 破損などの異常時のみ。

## Migration の原則

- `dls-core/manifest.json` の `migrations[]` に `id: YYYYMMDD_NNN` 形式で列挙する（Rails/Prisma スタイル）
- `.migration_state.json.last_applied_migration_id` より新しい ID のみ順次適用する
- アクション型は v1 で 6 種: `rename_file` / `delete_file` / `update_settings_hook_command` / `update_settings_fragment` / `replace_hook_command_prefix` / `update_settings_matcher`（DLS-139 / DLS-157）
- 冪等性を保つ（既に完了している操作は skip）
- 失敗時は `last_applied_migration_id` を更新せず、再実行で最初からやり直す

## 禁止事項

- `active.md` / `archive.md` / `raw/` の既存エントリを上書き・削除しない（追記のみ。忘却は archive への移動で表現する）
- `settings.local.json` のユーザー設定値を自動書き換えしない（通知プレースホルダ置換と DLS フック追加のみ）
- `.claude/CLAUDE.md` のマーカー外領域を自動書き換えしない
- `protected_paths` に含まれるファイル・ディレクトリを init.py / `/dls-update` / `/dls-recover` から変更しない

## active.md エントリ数の軟上限運用（DLS-144 / DLS-145）

active.md は **目安 30 件** で運用する。30 件超は禁則ではなく **減衰候補レビューの trigger**。
`docs/metadls/01-session-notes.md §24` ロードマップ Phase 集約期（複数 DLS が短期間で連続的に
確定する時期）の超過は許容する。

**減衰再評価（`skills/05_archive_management.md` Step 2.5）の trigger**:

1. **Phase 終了（一次 trigger）**: ユーザーが `tasks/next.md` / `tasks/todo.md` / コミット
   メッセージで「Phase X 終了」と明示宣言した時点。CC は自律判定せず宣言を待つ。
   各 Phase の完了条件は `docs/metadls/01-session-notes.md §24.9` を参照。
   本時点で完了条件確定済は Phase 1.5 のみ。他 Phase は未確定で、§24.13 後追い検証推奨事項
   とともに段階的に確定する。
2. **コミット集約イベント（CC が気づいたら提案する補助シグナル）**: 同一テーマの DLS が
   連続的にコミットされ、新提案が出ていない状態を CC が察知したとき、「Phase 完了候補です、
   減衰再評価しますか？」とユーザー確認を提案する（実行はユーザー承認後、CC が単独実行しない）。
   閾値の数字は持たない。

**警告境界**: `dls-post-active-md-check.py` は 25 件で注意喚起、30 件超で軟上限超過通知
（warnings カテゴリ）を出すが、いずれもブロッカーではない。
