---
paths:
  - ".claude/.dls/**"
---
# DLS: エントリ記述ルール

## whatフィールドのルール（IDD準拠）

```
良い例:
  what: セッションの有効期限を30分に制限する

悪い例（Howが混入している）:
  what: Redisをセッション管理に使い、TTLを1800秒に設定する
```

`what` は「外から見たときに何を満たすべきか」のみ。実装手段（ライブラリ、データ構造、処理順序）は含めない。

## 原本ストア（.claude/.dls/raw/）のルール

- 追記のみ。ファイルを削除・編集しない
- ファイル名: `YYYYMMDD_種別_概要.{ext}`（例: `20260323_hearing_login_flow.md`）
- 種別: `hearing`（ヒアリング）/ `email`（メール）/ `chat`（チャットログ）/ `doc`（ドキュメント）/ `discussion`（議論ログ。`/dls-discuss` skill が生成）

## 仮説の状態空間

仮説は4つの状態を遷移する。状態を取り違えると棄却ルールが破綻する。

| 状態 | 意味 | 遷移トリガー |
|---|---|---|
| **proposed** | 仕様検討時に挙がっている候補 | 仕様作成時に列挙 |
| **adopted** | 採用、実装に進む | 仕様時の選定 |
| **dormant** | 採用されなかった（保留） | adopted 確定の結果、自動でdormant化 |
| **rejected** | 能動的に「ダメ」と判定 | 検証イベント（テスト失敗 / ユーザー拒否 / 実装試行） |

**重要原則**:
- **dormant ≠ rejected**: dormantは将来再浮上しうる、rejectedは理由付きで「もう試さない」
- 棄却時に dormant 群を「機械的にゼロから再検討」しない
- 棄却事実を踏まえて dormant を再評価する（第3の道）
- 全 dormant が棄却される場合は仕様レベル supersede（want 自体を再定義）

## supersedes と rejected_hypothesis のセット

`supersedes` フィールドを書く場合は、**必ず `rejected_hypothesis` を併記する**（強く推奨）。

理由:
- `supersedes` だけでは「何の仮説を棄却したか」が読み取れない
- DRY チェックで CC が grep / ccc search して棄却済み仮説を検出できるようにする
- 同じ仮説の再提案（DLS-127 → DLS-128 型の撤回サイクル）を構造的に防ぐ

`rejected_hypothesis` のサブフィールド:
- `target`: 棄却対象の DLS ID（supersedes と重複可、明示性のため）
- `hypothesis`: 棄却された仮説の1行要約
- `reason`: 棄却の根拠（実証的: テスト失敗 / ユーザー拒否 / 実装試行など）

## commits フィールド: 関連 git コミットの記録（DLS-153）

DLS は判断のログだが、判断時のコード状態 / 反映差分 / 棄却試行の痕跡を構造化していない。
バックトラック（NG 仮説から手前に戻る）には「どの commit に戻ればその判断時点のコンテキストに
戻れるか」が必須情報。`commits` フィールドで構造化する。

サブフィールド:

| サブフィールド | 用途 |
|---|---|
| `baseline` | 判断時のコード状態 SHA。戻り先の起点 |
| `impl` | 判断を反映した実装コミット SHA（複数可、カンマ区切り）。戻す = revert する範囲 |
| `reject_evidence` | 棄却された案を実装試行した痕跡 SHA。dormant 再評価時の参照 |

適用範囲（段階導入）:
- **新規エントリ**（dls_core_version 1.14.0 以降で起票）: 判断とコードが直結する場合は少なくとも
  1 つのサブフィールドを記入する。全て任意だが、空のままだとバックトラック skill から不可視
- **既存エントリ**（1.14.0 未満で起票された 42 件）: 免除。protected_paths 配下のため migration
  では補完不可。遡及補完が必要ならユーザーが手動で active.md を編集する
- `/dls-update` は欠落を warning レベルで表示するが block しない

記入タイミング:
1. 判断確定時に `baseline: <現在 HEAD>` を記録
2. 実装コミット後に `impl: <SHA>` を追記
3. dormant 案を実装試行して NG なら `reject_evidence: <試行 SHA>` を追記し、
   `rejected_hypothesis.reason` で参照

`sources` フィールドとの役割分離: `sources` は raw/ 議論ノートへの参照（判断の根拠原本）、
`commits` は git コミットへの参照（判断のコード化痕跡）。

## 棄却を表現するエントリの what

仮説を棄却する DLS エントリは、`what` に「何の仮説を棄却したか」を明記する。

- ✅「<元の仮説を要約>を棄却し、〜とする」
- ❌「〜とする」（supersedes だけで仮説が読み取れない）

これにより、`rejected_hypothesis` の構造化検索と `what` のテキスト検索の二段構えで、新旧エントリ両方からの DRY チェックが可能になる。

## エントリ数が目安（30 件）を超えたとき

軟上限ルール（DLS-144）に従う。`.claude/.dls/skills/05_archive_management.md` の手順で
減衰候補レビューを実施し、Phase 終了 or コミット集約イベント時に整理する。
Phase 集約期の超過は許容され、機械的な archive 移動は強制されない。

## 想起と忘却の設計

DLSにおける **忘却 = archive.md への移動**。記憶を捨てるのではなく、活性度を下げる。

**強化シグナル（エントリが活きている証拠）:**
- コミット対象ファイルがエントリの `where` と重なる
- 他の新しいエントリが `depends_on` で参照している
→ 強化されたエントリは active.md に残し続ける

**減衰シグナル（archive候補）:**
- date が長期間前（プロジェクトの活動頻度に応じて判断）
- 他エントリの depends_on/affects から参照されていない
- git log で where のファイルが長期間変更されていない
→ 月次チェック（skill 02）で検出し、判断のうえ archive に移動する

## skill / hook / commands 新設時の CLI ハーネス依存の明示（DLS-151）

skill / hook / commands を新設する DLS エントリは、`where` または `assumption` に
**CLI ハーネス固有依存**（claude-code / codex / gemini のいずれに紐付くか）を明示する:

- frontmatter スキーマ（`name`, `description` 等の claude-code 規約）
- hook event 名（PreToolUse / SessionStart 等、各 CLI で差異あり）
- `.claude/` パス前提（codex は `.codex/`、gemini は `.gemini/`）
- ハーネス専用状態ファイル参照（`tasks/next.md` の `mode:` 行など）

これにより §24 Phase 3（codex / gemini 対応）着手時に adapter 設計の input として
grep 可能になる。Phase 3 着手前は明示するだけで足り、移植実装は不要。配布パスの
動作確証は PoC #1〜#4（DLS-129/130/131/132）で取得済み。
