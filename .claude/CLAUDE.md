<!-- DLS-CORE:BEGIN v1.0.0 -->
# CLAUDE.md — DLS運用ルール

このプロジェクトでは **DLS（Discovery Linked Specification）** を使用して判断を記録・追跡する。
コードを書く前に必ず `.claude/.dls/active.md` を読み込み、過去の判断を踏まえて作業すること。

DLSエントリは「ある時点での判断のログ」であり、仕様書ではない。
コードがどう変わっても過去のエントリは常に正しい。乖離という概念が存在しない。

- **エントリはキャッシュ**: 原本（sources）と表示の間の構造化キャッシュ。再生成可能。
- **原本は再生成不可能**: `sources` が消えたら DLS の長期的価値が失われる。
- **フォーマットは薄く保つ**: 必須フィールドを守り、それ以上追加しない。
- **単一候補は判断ではない**: 1つの候補・1つの視点だけでは採用判断は空転する。仕様検討時は実装候補を最低2案併置する。議論時は反対視点 / 「何もしない」候補を必ず1つ併置する。反対視点を作れない時は「作れない、なぜなら〜」と明言する（形だけの反対は禁止）。

## URL 本文取得（DLS-124）
特定 URL のページ本文を取得するときは `playwright-cli open <URL>` → `playwright-cli snapshot` を使う（公式 @playwright/cli、token-efficient）。Playwright MCP / Chrome MCP は重い副位置（CLI 不在時のみ）。セットアップは `docs/setup.md#playwright-cli`（`npm i -g @playwright/cli@latest`、init.py が不在検出時に提案）。クエリ検索は内蔵 WebSearch を使う（DLS-157 で `/gemini-search` 廃止）。

## コードベース意味検索（DLS-124）
「認証ロジックはどこ？」のような自然言語コード検索は `ccc search "<クエリ>"` を使う（cocoindex-code CLI、AST ベース、トークン 70% 削減）。初回は `ccc init` → `ccc index`。Grep より低トークンで実装探索ができるとき優先する。MCP は副位置。セットアップは `docs/setup.md#cocoindex`（`uv tool install --upgrade cocoindex-code --prerelease explicit`、init.py が不在検出時に提案）。

---

## 作業開始時
0. **`.claude/.dls/.migration_state.json` の `update_pending` フラグを確認する**
   - フラグが立っている場合: **他のいかなる作業よりも先に `/dls-update` を実行する**（init.py が dls-core を更新した直後の必須メンテナンス）
   - フラグは `/dls-update` 完了時に自動で降ろされる
1. `.claude/.dls/active.md` を読み込む
2. 今回の作業対象に `where` が重なるエントリを優先的に確認する
3. 関連エントリの `rejected_alternatives`（dormant 群）と `rejected_hypothesis`（能動的に棄却された仮説）を確認し、棄却済みの案を再提案しない

## モード別の応答態度

CC は以下のいずれかが成立したら「議論モード」に入る:
1. ユーザーが `/dls-discuss` または `/dls-plan` を呼んだ
2. Plan モード中である（ExitPlanMode ツールが利用可能）
3. ユーザーが「議論したい」「中立的に」「批判的に」「再検討」「検討」「相談」「比較」「選択肢」「どう思う」「妥当か」等のキーワードを使った

議論モード中の振る舞い:
- 反対視点を必ず1つ併置する（同意のみで終わらせない）
- 「何もしない」候補を必ず含める
- 賛成 / 反対 / 留保を明示する
- 反対できない時は「反対できない、なぜなら〜」と明言する（形だけの反対は禁止）
- 「全面同意」を出す前に追従バイアスを自己検出する

ユーザーが実装系シグナル（「実装して」「修正」「コミット」「動かして」等）に切り替えたら議論モードを解除し、執行集中（過剰な批判 / 反対視点を控える）に移行する。判定不能時は仕様モード寄りで対応する（反対視点を出す方がコストが低い）。

## 実装中に判断が発生したとき

**以下に該当する場合はDLSエントリを必ず作成する:**
- 技術的な選択肢から1つを選んだ
- バグの根本原因を特定した
- 外部制約により方針を変更した
- 性能ボトルネックを特定した
- 棄却した代替案がある
- 評価結果から設計方針を決定した

**判断基準**: 「3日後の自分がこの判断の理由を思い出せるか？」— 思い出せないなら記録する。

エントリ作成は `.claude/.dls/skills/01_generate_entry.md` の手順に従う。
**実装コミットの前に**エントリを追加する（判断時に即記録）。

**cross-project DLS 番号**: コミットメッセージ等で裸の `DLS-XXX` は本プロジェクトの `active.md` に存在する番号のみ。dls-core 配布物（skill / hook / adapter）由来の番号は `dls-core DLS-XXX` と明示して本プロジェクトの番号空間と混同させない。

## 変更リクエストを受けたとき
1. `.claude/.dls/skills/03_impact_analysis.md` で影響範囲を分析する
2. 既存エントリを上書き・削除しない。`supersedes` で新エントリを追加する
3. **`supersedes` を書く場合は `rejected_hypothesis` を併記する**（target / hypothesis / reason）
4. 上書きされたエントリは `archive.md` に移動する（＝忘却）

## 議論と仕様検討の skill

| 状況 | 使うもの |
|---|---|
| 議論したい（複数論点を比較・検討、結論を出して残す） | `/dls-discuss [topic]` |
| 仕様検討したい（want から複数仮説を列挙して 1 案を採用） | `/dls-plan [topic]` |
| 実装計画したい（採用案を todo に分解） | 通常の plan モード |
| codex 異モデルコードレビューを実行したい | `/dls-codex-review [--base/--commit/--dls DLS-XXX,...]` |
| フェーズ完了をコミット | `/dls-commit`（議論モード時は議論保存 + DLS 化 + モード解除も実行） |
| 次に何を実行すべきか迷う / コマンド一覧を見たい | `/dls`（引数なし=文脈推薦、`help`=全 dls-* の 3 列マトリクス） |
<!-- DLS-CORE:END -->
