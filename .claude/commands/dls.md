---
name: dls
description: 次に実行すべき dls-* コマンドを推薦するランチャー。引数 `help` で全 dls-* コマンドの使い分けマトリクスを表示する。
---

# /dls — DLS コマンドランチャー

引数なしで「次に何をすべきか」を 1 行推薦する。`/dls help` で全 dls-* コマンドの使い分けを 3 列マトリクスで表示する。

DLS-148 採用案。実装サイズを抑えることで DLS-143「skill 乱立回避」原則と共存させる。

---

## 引数

- なし → **推薦モード**
- `help` → **使い分けマトリクス表示**

---

## 推薦モード（引数なし）

以下を **並列で確認** する:

1. `.claude/.dls/.migration_state.json` の `update_pending` フィールド
2. `tasks/next.md` の「DLS 議論モード状態」セクション（`mode: discuss` / `mode: plan`）
3. `git status -s` の出力
4. `tasks/next.md` の存在

判定ルール（**優先順** に最初にマッチした 1 つを推薦）:

| 優先 | 条件 | 推薦 | 理由 |
|---|---|---|---|
| 1 | `update_pending: true` | `/dls-update` | dls-core 更新の必須メンテナンス（他作業より優先） |
| 2 | next.md に `mode: discuss` / `mode: plan` 行あり | `/dls-commit` | 議論モード解除 + 議論ノート保存 |
| 3 | `git status -s` 非空 | `/dls-commit` | 未コミット変更の集約 |
| 4 | `tasks/next.md` 存在 + 前セッション要約あり | `/dls-continue` | 前セッションの作業再開 |
| 5 | `tasks/next.md` 不在 or 内容空 | `/dls-summary` | プロジェクト全体把握から開始 |
| 6 | 上記すべてに該当しない | `/dls help` | 判定不能、使い分け表示 |

出力フォーマット:

```
次の推奨: /<command>
理由: （1 行の判定根拠）
```

---

## help モード（`/dls help`）

以下を 3 列マトリクスで表示する（コマンド名 / 用途 / いつ使うか）:

| コマンド | 用途 | いつ使うか |
|---|---|---|
| `/dls-continue` | 前セッションの作業を再開 | セッション開始時、`tasks/next.md` がある |
| `/dls-summary` | プロジェクト状況サマリーを生成 | 全体把握、進捗報告のベース |
| `/dls-discuss [topic]` | 議論モード起動 | 複数論点の比較・再評価・検討 |
| `/dls-plan [topic]` | 仕様検討モード起動 | want から複数仮説で 1 案を採用 |
| `/dls-commit` | フェーズ完了のコミット + next.md 保存 + モード解除 | 区切りのタイミング |
| `/dls-update` | DLS core 更新の適用 | `update_pending` フラグが立ったとき |
| `/dls-clean` | 共有 DLS データを承認付きで整理 | 整合性・todo DRY・archive を保守するとき |
| `/dls-recover` | 欠損ファイルの選択復元 | hook 削除 / settings 破損などの異常時 |
| `/dls` | コマンドランチャー（本コマンド） | 迷ったとき |

---

## ルール

- 推薦は **常に 1 つだけ**（複数候補は混乱の元）
- 推薦理由は **1 行**
- 判定不能なら `/dls help` を推薦して逃げ道を提供する
