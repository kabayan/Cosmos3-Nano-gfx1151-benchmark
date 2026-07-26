---
name: dls-update
description: DLS core 更新（配布状態、CLAUDE.md マーカー、エントリスキーマの同期と完了証明）
---
# /dls-update — DLS Core 更新

このコマンドは `dls-core/` の更新を既存プロジェクトに反映する。
`init.py`（新規セットアップ）とは違い、既に DLS が導入済みのプロジェクトを
最新版 dls-core に追従させることに特化する。

## 前提

- dls インストール先の `dls-core/manifest.json` が存在
- 本プロジェクトに `.claude/.dls/.migration_state.json` が存在
  （無ければ `init.py` を先に実行するようユーザーに案内する）

## 責務の明確化

- `/dls-update` は既存環境の更新に特化
- ファイル本体のコピー・マイグレーション適用は `init.py` 担当
- `/dls-update` は `init.py` 再実行を促し、その後マーカー同期とスキーマ移行を行う
- 整合性確認・todo 衛生・archive レビューは `/dls-clean` の責務であり、本コマンドでは実行しない

## Phase 0: dls リポジトリ更新確認

1. `.claude/.dls/.migration_state.json` を読み、`dls_path` フィールドを取得する
2. `dls_path` が未記録またはパスが無効な場合:
   - ユーザーに dls リポジトリのパスを質問する（例: `~/workspace/dls`）
   - 以降のステップでそのパスを使う（state への記録は `init.py` の責務なので行わない）
3. `git -C <dls_path> fetch` でリモートの最新を取得する
   - fetch 失敗時（ネットワーク不通等）は警告のみ表示し、Phase 1 に進む（ブロッカーにしない）
4. `git -C <dls_path> log HEAD..origin/master --oneline` で未取得コミットを確認する
5. 出力が空なら「dls は最新です」と報告し、Phase 1 に進む
6. 出力がある場合:
   - 更新内容（コミット一覧）をユーザーに提示する
   - 「dls を pull して更新しますか？」と確認する
   - Yes → `git -C <dls_path> pull` を実行し、Phase 1 に進む
   - No → 「現在のバージョンで続行します」と報告し、Phase 1 に進む

## Phase 1: バージョン確認

1. dls インストール先の `dls-core/manifest.json` から `dls_core_version` を取得する
2. 本プロジェクトの `.claude/CLAUDE.md` 内 `<!-- DLS-CORE:BEGIN vX.Y.Z -->` マーカーから
   現在バージョンを抽出する
3. 同一バージョンなら「更新不要」と報告して終了する
4. 新バージョンが古い場合はユーザーに警告して終了する

## Phase 1.5: 未有効 feature の検知（任意案内）

1. `manifest.features` を走査し、各 feature の `files[].condition.feature` 該当物が
   本プロジェクトに配布済みかを確認する
2. 未配布の feature があれば「この feature は未有効です。有効化するには
   `python init.py` を再実行してください」と**案内のみ**表示し、次フェーズに継続する
   （ブロッカー扱いにしない）
3. feature 検出と対話・配布は init.py の責務（DLS-114）なので `/dls-update` からは呼ばない

## Phase 2: CLAUDE.md マーカー同期（marker-bounded）

0. `.claude/CLAUDE.md` の存在を確認する。**不在の場合**は次の対応を取る:
   - 「`.claude/CLAUDE.md` が存在しません。init.py が初回作成を担当するため、
     `python <dls インストール先>/init.py` を再実行してください」と案内して終了する
   - Phase 2 以降には進まない（ファイル本体の作成は init.py の責務 / DLS-118）
1. `.claude/CLAUDE.md` を読み込む
2. `<!-- DLS-CORE:BEGIN ... -->` ～ `<!-- DLS-CORE:END -->` の区間を抽出する
3. マーカー未検出時の初回変換:
   - ユーザーに「既存 DLS セクションをマーカー付きに初回変換する」旨を確認する
   - 確認後、旧 DLS セクションをマーカー入りブロックに置換する
   - 変換に失敗したら「手動で marker を入れてください」と案内して終了する
4. マーカー検出時:
   - `dls-core/claude-md-fragment.md` で中身を置換する
   - マーカー文字列にバージョン番号を埋め直す（例: `<!-- DLS-CORE:BEGIN v1.0.0 -->`）
5. マーカー外のユーザー記述は絶対に変更しない

## Phase 3: エントリスキーマ移行

1. `.claude/.dls/.migration_state.json` から `last_applied_migration_id` を読む
2. `manifest.migrations[]` で `id > last_applied_migration_id` の未適用マイグレを列挙する
3. エントリ構造の変更を伴うマイグレがある場合のみ:
   - `.claude/.dls/active.md` の各エントリを `skills/01_generate_entry.md` の最新フォーマットと照合する
   - 欠損フィールドがあれば LLM 判断で候補を生成し、ユーザー承認後に追記する
4. 完了時に `.migration_state.json` を更新する（init.py が管理する migration state とは
   独立したエントリスキーマ version を持つ場合は別フィールドで管理する）

## Phase 6: update_pending フラグ解除（自動・必須）

DLS-122: init.py が dls_core_version 変更を検出して立てた `update_pending` フラグを降ろす。
`/dls-update` が Phase 0〜3 の必要な処理を正常完了した場合にのみ必ず実行する。
`/dls-clean` の実行、または Phase 1 で「更新不要」と判断しただけでは解除しない。

```bash
python3 -c "
import json
from pathlib import Path
p = Path('.claude/.dls/.migration_state.json')
if p.exists():
    state = json.loads(p.read_text(encoding='utf-8'))
    if state.pop('update_pending', None) is not None:
        p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('update_pending フラグを降ろしました')
    else:
        print('update_pending フラグは存在しませんでした（スキップ）')
"
```

これにより次回セッション開始時の SessionStart hook 警告は出なくなる。
途中フェーズで失敗・中断した場合はフラグを降ろさず、再実行を促す。

## エラー時の挙動

- Phase 2 で `.claude/CLAUDE.md` が不在なら init.py 再実行を案内して終了する（DLS-118）
- Phase 2 でマーカー変換に失敗したら Phase 3 以降に進まず終了する
- Phase 3 でエントリ欠損フィールドの自動生成に失敗したら該当エントリ ID を報告し
  次フェーズに進む（ブロッカー扱いにしない）
- 整合性・todo DRY・archive の確認が必要なら `/dls-clean` を明示起動する
