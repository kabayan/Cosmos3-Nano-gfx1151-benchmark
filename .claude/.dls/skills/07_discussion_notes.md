# Skill 07 — 議論ノートの抽出と保存

セッション中の設計議論・中立的再評価・YAGNI 判定・外部事例参照などを、
DLS エントリの `sources` を補強する原本として `.claude/.dls/raw/` に保存する。

DLS エントリ単体では「結論と棄却案」しか残らない。**結論に至るまでの議論の構造**
（採用した評価軸、外部事例の参照、抽象化レイヤーの議論、未解決の論点）は
別途記録する価値がある。これが 3 ヶ月後・他者・自分自身の再現性を支える。

---

## 原則

1. **議論ノートは原本（raw/）に置く**: 追記のみ、削除・編集しない（CLAUDE.md / dls-entry.md の原本ストア規約）
2. **DLS エントリは結論のキャッシュ**: 議論ノートが原本、DLS は再生成可能なエントリ
3. **DRY**: 議論ノートと DLS の `what` / `why` / `rejected_alternatives` を重複させない。議論ノートは「結論の前段」を残し、結論自体は DLS にリンクする

---

## 抽出対象

以下のいずれかに該当する議論があった場合、議論ノートを作成する候補:

- **設計探索**: 機能の必要性・スコープを議論し、minimal slice / YAGNI / 中立的再評価を経たもの
- **外部事例参照**: 他ツール（Codex / Aider / Cursor 等）との比較
- **抽象化レイヤーの議論**: 3 層モデル / WBS 相関 / 粒度判定など
- **却下案の理由整理**: 採用しなかった選択肢を後で見返すための記録
- **仮説提示**: 検証はまだだが、後で実機検証する価値がある仮説

**判断基準**: 「**3 ヶ月後にこの議論の構造を再構築できるか**」 — 再構築困難なら抽出。

---

## 抽出対象外

以下は議論ノートを作らない:

- 単発の質問応答（DLS エントリの `sources` フィールドに 1 行書けば十分）
- 実装の選択（DLS エントリの `rejected_alternatives` で十分）
- 進捗報告（next.md / コミットメッセージで十分）
- バグ修正の議論（DLS エントリの `why` で十分）

---

## 保存先とファイル名

### 議論ノート（議論単位）

- 場所: `.claude/.dls/raw/`
- ファイル名: `YYYYMMDD_chat_<topic>.md`
- 種別: `chat`（dls-entry rules で定義済み: `hearing` / `email` / `chat` / `doc`）
- 例: `20260511_chat_m3-design.md` / `20260612_chat_auth-rewrite.md`

### jsonl アーカイブ（session 単位、DLS-022）

議論ノートと**セット**で、対応する cc transcript jsonl を zip 圧縮して保存する:

- 場所: `.claude/.dls/raw/jsonl/YYYYMMDD/`（日付別サブフォルダ、ファイル数増対策）
- ファイル名: `<session_id>.zip`（cc session 単位、`<full-uuid>.zip`）
- 例: `.claude/.dls/raw/jsonl/20260511/93e9820f-....zip`

**重要**: 1 session の中に複数議論ノートが生まれることがある（例: 同じ cc セッションで M3 議論 + 議論ノート運用議論）。粒度の違い:
- **議論ノート**: 議論単位（複数 / session 可）
- **jsonl アーカイブ**: session 単位（1 / session）

議論ノートのメタ部に `session_id` を明記して jsonl アーカイブにリンクする。

---

## ノートのメタフィールド（必須）

議論ノート本文の冒頭に以下のメタ情報を必ず含める:

```markdown
# <タイトル>

> 作成日時: YYYY-MM-DD HH:MM
> 出典: cc session `<short-id>` / jsonl archive `<.claude/.dls/raw/jsonl/<date>/<full-id>.zip>`
> session_id: `<full-uuid>`
> 関連 DLS: DLS-NNN, DLS-MMM
```

`session_id` を起点に、後から jsonl から表・コード等のアーティファクトを抽出できる。
`関連 DLS` で双方向検索を支える。

---

## ノートの構造（推奨セクション）

議論内容に応じて以下から選択。**必須ではない**:

1. **出発点** — 観測された痛み、外部からの要求、議論のトリガー
2. **外部事例参照** — 他ツール・他プロジェクトの先行事例（該当時のみ）
3. **中立的再評価** — 採用した評価軸（完全性 / YAGNI / 他ツール適用性 等）
4. **確定事項** — minimal slice、却下案、棚上げ事項
5. **関連 DLS エントリ** — 議論から起きた / 強化した DLS 番号
6. **議論再開時の起点** — 未解決の論点 + 推奨アプローチ
7. **検証根拠（実機があれば）** — コマンド出力、実機検証のスナップショット

---

## DLS sources への追記

議論ノート作成後、対応する DLS エントリ（active.md）の `sources` フィールドに
リンクを追記する:

```markdown
- **sources**: 既存 source 1, ..., `.claude/.dls/raw/YYYYMMDD_chat_<topic>.md`
```

複数の DLS エントリが同じ議論を参照することは正常。同じ raw ファイルへの
リンクを各 DLS の `sources` に追記してよい（双方向検索を支える）。

逆方向（議論ノート → DLS）は議論ノート本文の「関連 DLS エントリ」セクションで表現する。

---

## /dls-commit Phase 1.6 から呼ばれる手順

⚠️ **重要**: 議論ノートが消失すると **復活できない**。会話コンテキストは `/clear` / cc 再起動で失われるため、議論ノートが唯一の原本。判定は **作成側に倒す**（false negative < false positive）。

1. **抽出判断（疑わしいなら作成）**:
   - 設計議論・中立的再評価・外部参照・抽象化議論のいずれかの兆候が**少しでもあれば**抽出対象とする
   - 判定に迷う場合は **必ず作成側に倒す**（軽すぎる議論ノートは後で archive すればよいが、消失した議論は復活不能）
   - 「議論なし」と確信できる場合のみスキップ可能（例: バグ修正のみ・進捗報告のみのセッション）

2. **判断結果を必ず明示**:
   - 抽出する場合: ドラフト提示 + ファイル名 + 関連 DLS リンク
   - スキップする場合: **「議論なし」の判断理由を明示**（例: 「本セッションはバグ修正のみで設計議論はなかった」）
   - 黙ってスキップしない（ユーザーが判定を再評価できる材料を残す）

3. **該当ありなら**:
   1. ファイル名候補を提案（`YYYYMMDD_chat_<topic>.md`、topic は議論のキーワード）
   2. ノート本文のドラフトを作成（メタフィールド必須 + recommended sections から議論に合うものを選ぶ）
   3. **ユーザーに確認**: ファイル名・本文・関連 DLS エントリへの sources 追記を提示
   4. 承認後、`.claude/.dls/raw/` に保存
   5. **jsonl アーカイブを作成**（DLS-022）:
      - cc 依存環境では transcript アーカイブスクリプト（ksd では `bin/cc-archive-jsonl <session_id>`）を実行
      - `.claude/.dls/raw/jsonl/YYYYMMDD/<session_id>.zip` が生成される
      - 同一 session の jsonl は冪等（既存なら skip）
   6. 関連 DLS エントリ（active.md）の `sources` にリンクを追記（議論ノート + jsonl アーカイブ両方）
   7. コミット対象に含める（`git add .claude/.dls/raw/<file>` + `git add .claude/.dls/raw/jsonl/<date>/<session>.zip` + `git add .claude/.dls/active.md`）

4. **/clear / cc 再起動の前に必ず実行する**: 二段防衛（DLS-146）で議論ノート未作成を検知する。Stop hook（`dls-stop-reminder.py`）が CC 応答完了ごとに **予防警告** を出し、SessionEnd hook（`dls-session-end-reminder.py`, matcher: `"clear"`）が /clear 直前に **最終確認警告** を出す。いずれかの警告が出たら必ず /dls-commit を経由してから /clear する。

5. **jsonl アーカイブの取り扱い**:
   - jsonl の原本（`~/.claude/projects/<encoded-cwd>/<session>.jsonl`）は cc 標準管理下、削除されないが保証なし
   - **議論ノート作成と同時に jsonl をアーカイブ**することで、cc 削除されても git で保護される
   - 1 session に複数議論ノートある場合、jsonl アーカイブは 1 回で済む（冪等性のため再実行 OK）
   - 後から jsonl から表・コードを抽出: `unzip -p <archive>.zip <session>.jsonl | jq ...` で取り出し可能

---

## 注意事項

- **自動保存はしない**: ドラフトを作って **必ずユーザー確認を経る**。質の悪い議論ノートが raw に積もると検索性が落ちる
- **議論ノート内のリンクは絶対パスでなく相対パス**: `path/to/file:42` 形式、コミット内 sha は 7 桁短縮可
- **既存議論ノートは編集しない**: 続きの議論があれば新ファイル（`YYYYMMDD_chat_<topic>-followup.md`）として追記
- **議論ノートと DLS エントリの寿命**: DLS エントリは active → archive へ動く、議論ノートは raw に永続化（archive されない）
- **ksd / wcc / dls など複数プロジェクトを跨ぐ議論**: 主導プロジェクトの raw/ に置き、関連プロジェクトの DLS から sources でリンクする
