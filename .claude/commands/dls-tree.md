---
name: dls-tree
description: dls-plan が残した複数提案を frontier 化し、各ノードを active.md に dormant 投影（projection-not-store）して tier 判定を付け、dls-exec に渡す。新永続ストアは作らない。明示起動時のみ実行（暗黙起動しない）。
---

# /dls-tree — 提案群の frontier 化（DLS projection）

入力 = dls-plan が列挙した複数提案。出力 = frontier（DLS active.md の projection）。
提案を「探索すべきノード群」として active.md に dormant で記録し、tier 判定を付けて
dls-exec に渡す。**tree は永続実体でなく DLS の projection**（第4ストアを作らない）。

**この skill の存在理由 = projection-not-store**: frontier は専用ファイルでなく、
DLS の dormant/rejected エントリとして表現する。探索状態の永続化先は active.md ただ一つ。

> 🛑 **主（メイン）セッションで起動する。ノード探索を subagent 並列化しない**（DLS-164）:
> 後続の dls-exec が依拠する E2 hook（定量 verdict ゲート）は subagent 文脈で空洞化する
> （subagent の PreToolUse payload が親セッション id を持ち、親側 Bash を誤って数える＝E2-glob で実証）。
> dls-tree が frontier ノードを subagent に分配して並列探索させると、各 subagent 内で gate が効かない。
> 並列化が必要になったら hook の subagent 対応 or 別 enforce 機構が前提条件（本ドラフトの射程外）。

## 手順（STEP A–D）

### STEP A: COLLECT
dls-plan の出力（直前の応答 or 議論ノート `raw/*_discussion_*.md` / `raw/*_chat_*.md`）から
**提案群を読み取る**。各提案について「考え方の要約」と「採否を分ける指標」を 1 行で抽出する。
- 提案が 1 件しか無いなら dls-tree は不要（dls-exec に degenerate 1 ノードで直接渡す）。
- 提案が複数なら frontier 化に進む。

### STEP B: PROJECT（projection-not-store の核 / frozen-log 整合）
各提案を frontier ノードとして **in-context で保持**する（§3 の volatile churn と同扱い）。
**active.md への記録は dls-exec の決着後に frozen log として一度だけ行う**（未決定の判断エントリを
先に書いて verdict ごとに annotate するのは DLS の append-only / frozen-log 不変条件と衝突するため。
dogfood #4 / DLS-167 で確定）。決着時の frozen log の形:
- 判断本体エントリが無ければ **新規 DLS を起票**し、**採用候補を `what`、未採用の残り提案群を
  `rejected_alternatives` に dormant、探索して棄却した提案を `rejected_hypothesis`** に記録する
  （`01_generate_entry.md` の手順）。
- 判断本体エントリが既にあれば、その `rejected_alternatives` / `rejected_hypothesis` に追記する。
- **新永続ストア（tree.md / frontier.json 等）を作らない**。frontier の永続表現は決着後 active.md の
  エントリ **のみ**（projection-not-store）。
- 揮発 churn（探索中の中間状態・frontier 全体）は **専用ファイルに書かず in-context で保持**する（§3 参照）。

### STEP C: CLASSIFY tier（既定 enforce = F2）
各ノードに **tier 判定**を付ける（dls-exec STEP 3 = CLASSIFY と同じ規律）。
- **疑わしきは定量扱い**: ノードの採否に数字が 1 つでも絡むなら定量 node とする。
- 誤って定量扱いしても害は「dls-exec で念のため code 実行」だけ（F2、安全側）。
- tier の自動分類ルータ（F1）は作らない（Refinement）。

### STEP D: HANDOFF
frontier を dls-exec に渡す。
- **渡す形式 = active.md の dormant エントリ群そのもの**（dls-exec STEP 1 SELECT は
  active.md projection から frontier を構築する）。別ファイルでの受け渡しはしない。
- best-first の事前順位（人間 / 仕様の優先度）を STEP B のノード記述に 1 行添える
  （dls-exec が SELECT の根拠に使う。BO エンジンは Refinement）。

## やらないこと（Refinement / 重い版 — spec §6）
- 明示 3 階層 tree（flat frontier で足りる、フォーク1=B）
- tier の LLM 分類ルータ（フォーク2 案 F1）
- 揮発 frontier の専用永続ファイル（projection-not-store。§3）
- BO / 期待値エンジン / fidelity ladder 明示 / 意味検索 dedup
