---
name: dls-recover
description: DLS core の緊急復旧（マニフェスト突合、欠損ファイル復元、settings 再マージ）
---
# /dls-recover — DLS Core 緊急復旧

通常運用では使わない。以下のような異常状態からの復旧に限定する。

- `.claude/hooks/` の一部が削除された
- `.claude/settings.local.json` から DLS フックエントリが消失した
- 手動編集で `.claude/` 配下の構造が破損した

`init.py`（新規セットアップ）や `/dls-update`（バージョン追従）とは責務が異なり、
「既存プロジェクトの欠損を対話的に埋め戻す」ことに特化する。

## 前提

- dls インストール先の `dls-core/manifest.json` が存在する
- 本プロジェクトが過去に `init.py` を実行済み（そうでなければ `init.py` を先に案内する）

## 責務の明確化

- `/dls-recover` は欠損検出と個別復元に特化する
- マイグレーション適用・マーカー同期は行わない（`init.py` / `/dls-update` の責務）
- `active.md` / `raw/` / `archive.md` / `tasks/*` は絶対に触らない

## Phase 1: マニフェスト突合（reconciliation）

1. `dls-core/manifest.json` を読み込む
2. `files[]` の各 `dst` が本プロジェクトに存在するか確認する
3. 欠損ファイルを列挙し、各ファイルに「復元 / スキップ」を対話で選択させる
4. 復元対象は `dls-core/<src>` からコピーする（`manifest.mode` を適用）
5. 保護対象パス（`manifest.protected_paths`）に該当する欠損は復旧しない
   （配布物ではないため `init.py` と同じルールを踏襲する）

## Phase 2: settings_fragments 再マージ

1. `.claude/settings.local.json` を読み込む
2. `manifest.settings_fragments.hooks` を走査する
3. `(event, matcher, command_basename)` の三つ組で重複を判定し、
   本プロジェクト側に無いエントリのみ追加する（上書き・削除はしない）
4. 通知系 curl エントリ等の L2 固有設定には一切触らない
5. basename 比較は `"/" + basename` でパス境界まで含めて照合する
   （部分文字列ヒットによる誤検出を避けるため — DLS-109 参照）

## Phase 3: protected_paths 保全確認

1. `active.md` / `raw/` / `archive.md` / `.migration_state.json` 等の存在を確認する
2. 欠損を検知しても復旧はしない（配布物ではないため）
3. 欠損がある場合は警告のみ表示し、ユーザーに手動復旧を促す

## Phase 4: 整合性検査（読み取りのみ）

1. `.claude/.dls/.migration_state.json` の存在を確認する
2. 無ければ `null` 初期化で新規作成する（既存があれば触らない）
3. 未適用 migration は再実行しない。必要なら `/dls-update` を案内する

## 禁止事項

- `active.md` / `raw/` / `archive.md` / `tasks/*` を絶対に上書きしない
- ユーザー確認なしにファイルを削除しない
- `settings.local.json` の既存エントリを削除・書き換えしない（追加のみ）
- マイグレーションを再実行しない（冪等性は `init.py` が担保する）

## 成功条件

- `manifest.files` の全 `dst` がプロジェクトに存在する
- `settings.local.json` に DLS フックの必須エントリが揃う
- 最終レポートで「復元したファイル一覧」と「スキップしたファイル一覧」を提示する

## エラー時の挙動

- Phase 1 で `dls-core/<src>` が不在の場合は該当ファイルの復元を諦め、
  ユーザーに dls 側の状態確認を促す
- Phase 2 の JSON パースに失敗した場合は読み取り専用で終了する
- Phase 3 の欠損は致命度に応じてユーザーに判断を委ねる
