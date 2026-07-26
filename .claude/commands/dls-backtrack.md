---
name: dls-backtrack
description: 過去判断 (goal-tree) に戻る薄い「二 tree 橋渡し」前段。戻り先 DLS を locate し (C-6)、実装 tree を commits.baseline へ branch 復元 (assist)、/dls-plan に handoff して再探索する (D-2)。探索エンジンは持たない (dls-exec に内包)。主経路は exec/plan/discuss の行き止まりからの handoff (内部 suggest)、副経路が明示起動。backtrack 経過 (goal-tree のみか実装-tree=git も伴うか) を DLS に残すのが主目的。
---

# /dls-backtrack — 二 tree 橋渡しの薄前段

入力 = 「過去の判断 X に戻ってやり直したい」という意図（topic 引数 / `DLS-XXX` / 引数なし）。
出力 = 戻り先 goal-tree node の確定 + 実装 tree を baseline へ branch 復元（assist）+ /dls-plan への handoff。

**この skill の存在理由 = 二 tree 橋渡し**: 探索エンジンを新設しない（DRY / projection-not-store）。
goal-tree（判断 / 仮説の木 = DLS 層）の再探索は dls-plan→dls-tree→dls-exec に内包される。
固有責務は (1) cross-decision の戻り先 locate（exec の SELECT は今の小 frontier しか見ない）、
(2) 実装 tree（コード状態 = git 層）の同期、(3) 再探索レールへの handoff の 3 点のみ（DLS-168）。

**主目的 = observability**: backtrack の主眼は「戻る機構」でなく **backtrack 経過を DLS に残すこと**。
(痛み1) 暗黙の backtrack（候補 A を実施→別候補 B へ移行）が残らない / (痛み2) 「戻して」が git で戻すのか
DLS だけで戻すのか混在し、どちらだったかが残らない——を解く。**この skill が触る判断は必ず
「goal-tree のみ」か「goal-tree + 実装-tree=git」かを明示して記録する**。

> 🚦 **起動経路（主 = 内部 suggest、副 = 明示。暗黙起動しないとは言わない）**:
> backtrack は正準サイクルの例外エッジ。**主経路** = exec 3段目（goal リーフ全滅）/ plan・discuss が「過去フォークに
> 戻る必要」を検出したとき、本 skill を **差し出す**（自動実行しない）。**副経路** = 人間発の cross-decision
> step-back に `/dls-backtrack <topic|DLS-XXX>`。**git 復元の実行は trigger 非依存で必ず人間承認**（assist）。

> 🛑 **主（メイン）セッションで起動する**。handoff 先の dls-exec が E2 hook の主セッション制約を持つ（DLS-164）。

> ⚠️ **git 復元部（STEP 2）は当リポでは dogfood 不能な既知限界**: 当リポは実装 tree が薄く
> （判断の "実装" が doc/DLS エントリでコードモジュールでない）、commits.baseline の checkout が
> 差分を生まない。STEP 1（locate）+ STEP 3（handoff）は当リポで dogfood 可能。STEP 2 は
> **実装 tree が thick なプロジェクト向け**で実機検証はそこで行う（DLS-164 subagent 限界同梱の precedent / DLS-168）。

## 手順（STEP 1–3）

**冒頭で backtrack 種別を判定する（痛み2 = 「戻して」二義性の直接解）**: 入口（特に「戻して」自然言語）で
**(i) goal-tree のみ**（DLS supersede / dormant 再活性、コード復元なし）か **(ii) goal-tree + 実装-tree=git**
（branch 復元も伴う）かを分類する。(i) なら STEP 2 を skip し STEP 1→3 のみ。**どちらだったかを最終記録に明示**
（決着 DLS の `commits` に git 復元有無 / `reject_evidence` の有無で表現）。曖昧なら人間に確認する。

### STEP 1: LOCATE（goal-tree、C-6 ハイブリッド）
戻り先の過去判断を**発見コスト最小**で特定する（論点 C = C-6、DLS-166）。
- **入力 2 系統（ハイブリッド）**:
  - `DLS-XXX` 直接指定（番号既知）→ そのエントリに直行（C-4 path、精密）。
  - topic 文字列（`/dls-backtrack <topic>`）→ active.md + archive.md を corpus に grep/parse し、
    `what` / `where` / `rejected_alternatives` を topic で照合して候補列挙（C-1 path）。
  - 引数なし → next.md / 直近文脈から戻り先を推測して候補提示（C-2 path、補助）。
- **threshold 絞り込み（C-6 の核）**: 候補が **10 件超**なら全件提示せず、topic / where での再絞り込みを促す
  （corpus は active+archive で 69 件規模 = 全件提示は認知負荷過大、DLS-166 computed）。10 件以下なら
  一覧提示してユーザーに 1 件選ばせる。threshold=10 は **tunable default**（最適値は invocation 履歴なく
  未検証 = DLS-166 assumption (2)）。
- **出力**: 戻り先 DLS-XXX 1 件 + その `commits.baseline` SHA（無ければ「baseline 未記録 = git 復元不可」と明示）
  + `sources`（raw/）の逆引き表示（論点 A-3 = **DLS + git + raw クラスタ**を一括提示）。

### STEP 2: SYNC 実装 tree（git assist、論点 A 安全形 = branch）
戻り先の `commits.baseline` へ実装 tree を **branch で**復元する。**assist に留める**（破壊的操作は人間判断、DLS-168 制約 (1)）。
- **現作業（NG 実装）の保全を最初に行う**（戻る前の状態を失わない）:
  - 未コミット変更があれば commit を促す（or stash）。
  - 現 HEAD（= NG 実装の到達点）SHA を保全 ref として残す。**この SHA が後で `reject_evidence`**。
- **戻り先 branch を切る**（hard reset しない）:
  - 命名: `backtrack/DLS-XXX`（戻り先判断 ID を主キー = grep 可能。衝突時は `-2` 等を付す）。
  - `git branch backtrack/DLS-XXX <baseline-SHA>` → `git switch` を **ユーザー承認の上で**実行
    （assist = 提案 + 承認実行まで。hard reset / 強制切替はしない）。
- **NG 実装 ref の記録経路**: 保全した現 HEAD SHA を **handoff payload に乗せる**。この skill 自身は
  active.md を書かない（frozen-log は決着時に dls-plan/exec 側が、戻り先を supersede する新エントリの
  `commits.reject_evidence` に記録する。dls-tree STEP B と同じ「決着後 frozen write」規律）。
- **当リポ既知限界**: 上記は実装 tree が thick な前提。当リポ（薄実装）では baseline checkout がコード差分を
  生まないため、この STEP は skip され locate → handoff のみ走る。

### STEP 3: HANDOFF（D-2 = /dls-plan 引き継ぎ）
再探索レールに接続する（論点 D = D-2、DLS-166）。
- 戻り先 DLS-XXX の **`rejected_alternatives`（dormant 群）を seed** として /dls-plan に引き継ぐ
  （「この判断の未採用案から再探索したい」）。/dls-plan → dls-tree → dls-exec が再探索を担う（探索は内包）。
- **二重起票の防止**: 再探索の決着は戻り先 DLS-XXX を **supersede する新エントリ**として書く（新規乱立させない）。
  NG 実装があれば新エントリの `rejected_hypothesis` + `commits.reject_evidence`（STEP 2 で保全した ref）に記録。
- backtrack が些細で /dls-plan が重すぎる場合の軽量フォールバック（D-1 = 新規 DLS テンプレ直接提示）は
  当面持たない（YAGNI、DLS-166 で D-2 のみ採用）。

## やらないこと（Refinement / 重い版）
- 別の backtrack 探索エンジン（探索は dls-plan→dls-tree→dls-exec に内包、DRY）。
- git の auto hard reset / 強制巻き戻し（喪失・衝突リスク。安全形は branch + 人間判断、論点 A / DLS-168）。
- 新永続ストア（frontier.json / backtrack-log 等）。frozen-log は決着時に active.md エントリのみ（projection-not-store）。
- threshold 値の自動チューニング（invocation 履歴が貯まるまで固定 default、DLS-166 assumption (2)）。
- codex / gemini 移植（claude-code skill = `.claude/` パス前提。Phase 3、DLS-151）。
