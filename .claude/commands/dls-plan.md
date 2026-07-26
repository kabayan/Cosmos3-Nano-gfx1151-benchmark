---
name: dls-plan
description: 仕様検討モード起動。「仕様 → 仮説」プロンプトを強制する。want から複数仮説を列挙し 1 案を採用。既存 DLS の DRY チェック（特に棄却 DLS）を強制。実装計画は通常の plan モードを使う。
---

# /dls-plan [topic] — 仕様検討モード

「仕様 → 仮説」のプロンプトを強制する仕様検討モード。
通常の plan モード（実装計画用）とは役割が異なる:

| 場面 | 使うもの |
|---|---|
| 仕様検討（複数仮説から選ぶ） | **`/dls-plan`** |
| 実装計画（採用案を todo に分解） | **通常の plan モード** |

`/dls-plan` 完了後、ユーザーが手動で通常 plan モードに切替えて実装計画を立てる
（自動切替はしない）。

---

## Step 1: want の確定

- 仕様（外から見て何を満たすか）を 1 行で書く
- **How（実装手段）は禁止**

良い例: 「セッションの有効期限を 30 分に制限する」
悪い例: 「Redis を使い TTL を 1800 秒に設定する」（How 混入）

---

## Step 2: DRY チェック（強制）

以下を並列で実行し、過去の関連判断を確認:

- `grep -l "<topic キーワード>" .claude/.dls/active.md .claude/.dls/archive.md`
- `ccc search "<topic>"`（cocoindex-code、ある場合）

**棄却 DLS を優先確認**:
- 各エントリの `rejected_hypothesis` を読む
- 「DLS-Y で棄却済み: <hypothesis>: <reason>」があれば候補から除外
- `rejected_alternatives`（dormant 群）も確認し、再評価可能性を判断

検出した dormant / rejected を要約してユーザーに提示。

---

## Step 3: 候補（仮説）列挙

- **最低 2 案を併置**（単一候補は判断ではない、claude-md-fragment.md 基本原則）
- 各候補の trade-off を 1 行
- 棄却済みの方向は避ける（Step 2 で検出したもの）
- **「何もしない」も候補に含める**（YAGNI 適用）

1 案しか出ない場合は **立ち止まる**: 既存 DLS の `where` 類似を再確認し、
本当に他案がないか検討する。

---

## Step 4: 採用と dormant 化

- 1 案を採用、残りは dormant
- 各 dormant に「採らなかった理由」を 1 行で記録（rejected_alternatives へ）
- **検証手段を決める**:
  - テストがある → そのテストが基準
  - テストがない → 「現状こう」観測（実装後に使われている事実が弱い裏付け）

---

## Step 5: DLS エントリ化（必須）

採用判断を DLS エントリとして保存する。
templates/entry.md に従い、以下を必ず記入:

- `what`: 採用案の仕様（How 禁止）
- `why.origin`: user_request / user_confirmed / implementation
- `where`: 影響範囲
- `sources`: 関連する raw/discussion ファイル（あれば）
- `rejected_alternatives`: dormant 群（Step 4）
- `assumption`: 未検証の前提（confidence: high/medium/low）

過去の判断を覆す場合は:
- `supersedes`: 旧 DLS-XXX
- **`rejected_hypothesis`** を併記（target / hypothesis / reason）
  → rules/dls-entry.md「supersedes と rejected_hypothesis のセット」参照

---

## Step 6: 解除（/dls-commit に委譲）

このコマンドは解除フローを持たない。`/dls-commit` を呼ぶと:
- DLS エントリが active.md に確定される
- 議論ノートがあれば skill 07 経由で raw に保存
- 仕様モード解除 + git commit + tasks/next.md 更新

専用の `/exit-plan` は不要。`/dls-commit = フェーズ完了` の意味付けに統一。

---

## 注意

- `/dls-plan` は仕様検討専用。実装計画は通常 plan モード
- 1 案しか出せない場合は YAGNI（「何もしない」）も含めて 2 案にする
- 棄却 DLS の rejected_hypothesis を見落とすと撤回サイクル（DLS-127 → DLS-128 型）が再発する
