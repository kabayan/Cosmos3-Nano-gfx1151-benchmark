---
name: dls-exec
description: dls-tree の frontier を best-first で探索し、各 node を試行・採点・backtrack する。定量 node は code 実行の捕捉出力なしに verdict を記録できない（INV-Q）。明示起動時のみ実行（暗黙起動しない）。
---

# /dls-exec — frontier 探索 + 定量ゲート

入力 = dls-tree が出した frontier（DLS active.md の projection。小タスクは 1 ノード）。
各イテレーションで 1 ノードを試行し、frontier を絞り、枯渇したら DLS に理由付きで
rejected/dormant を書き戻して backtrack する。

**この skill の存在理由 = INV-Q**: 定量 node の verdict は、数値を産んだ実行 code の
捕捉出力を伴わなければ STEP 6 で記録を拒否される。暗算 verdict は通さない。

> ⚠️ **機械強制（E2 hook）が併走する**: prose（本 SKILL.md）の INV-Q は cheap モデルに
> binding でない（P1 検証で haiku が `computed_from:` を捏造）。真の不変条件は「当該応答ターンで
> 実際に code 実行ツール（Bash）が invoke されたか（tool_uses>0 の観測）」であり、これは
> `.claude/hooks/dls-pre-quant-verdict-gate.py`（PreToolUse）が out-of-band で強制する。
> prose を踏み越えても hook が `computed_from:` 付き write を Bash 不在ターンで block する。
> rigor を回すモデルは capable（sonnet 以上）を推奨（P1 で capable は 6/6 実実行・捏造ゼロ）。

> 🛑 **この skill は主（メイン）セッションで起動する。subagent に委譲しない**（DLS-164 / dogfood #3）:
> E2 hook は「現ターンの Bash」を transcript 走査で数えるが、**subagent の PreToolUse payload は
> 親セッション id を持つ**ため、hook は subagent でなく**親（メイン）側ターンの Bash を誤って数える**。
> 結果、subagent 内では gate が空洞化し、Bash ゼロの捏造 verdict も素通りする（E2-glob で実証）。
> 将来 dls-tree がノード探索を subagent 並列化する場合も同様に **gate 非カバー**になる点に注意
> （その時は hook 側の subagent 対応 or 別 enforce 機構が必要）。メイン起動なら gate は正常
> （dogfood #1/#2 実証）。

## イテレーション手順（STEP 1–7）

### STEP 1: SELECT
frontier から best-first で 1 ノードを選ぶ。期待値の出所は thin では「人間 / 仕様の事前順位」で足る
（BO エンジンは Refinement）。

### STEP 2: SURFACE
そのノードの判断に必要な **生データを state に必ず露出する**（生カウント・分布・独立 split の存在）。
- evidence-skepticism（同音/交絡仮説の反証）と独立-split trigger は、生データさえ surface すれば
  capable モデルが naked でも自発発火する（enforce 不要）。これが (c)。
- **独立検証 set はここで使い切らない**。「最初の探索」で温存し、採用候補の検証に回す。

### STEP 3: CLASSIFY（既定 enforce = F2）
このノードの判断が「数値が結論を左右するか」を判定する。
- **疑わしきは定量扱い**: 判定に数字が 1 つでも現れたら既定で定量 node とする。
- 誤って定量扱いしても害は「念のため code 実行」だけ。安全側に倒す（F2）。

### STEP 4: COMPUTE（INV-Q の発生源）
定量 node なら、**判断に進む前に code を書いて実行し、その stdout を捕捉する**。
- baseline / null との比較を **必ず code 内で出す**（差・比・p 値など、結論を左右する量）。
- 暗算・概算・記憶した式の口頭適用は **禁止**。captured output を持たない数値は STEP 6 を通れない。
- 捕捉した出力を verdict の `computed_from:` に貼る（次段で必須チェックされる）。

### STEP 5: DECIDE
採点する。
- **tier1（機械評価）**: 機械的に測れる量（スコア・通過数・誤差）でそのまま採点。
- **tier2（validity ゲート）**: proxy（自動指標 / LLM judge 等）が goal を妥当に encode するか先に確認してから
  採点する。tier2 固有の危険（proxy-validity / generalization / signal-noise 混同）は code では捕まらないが、
  STEP 2 で confound を surface すれば naked で自発発火する（(c)）。
- 数値の値そのものより「proxy が goal を測れているか」を疑う（tier2）。

### STEP 6: WRITE（INV-Q ゲート — 記録段で拒否）
verdict を frontier に反映する。**記録の前に次をチェックする:**

```
IF ノードが定量 node（STEP 3 で定量判定）:
    IF verdict.computed_from が空 / 暗算由来 / captured output でない:
        → 記録を拒否する。STEP 4 に戻って code を実行する。
    ELSE:
        → verdict を computed_from 付きで記録する。
```

verdict 記録の必須フィールド（定量 node）:
- `verdict:` 採否（採用 / 棄却 / 保留）
- `computed_from:` STEP 4 で捕捉した実行 code の出力（**空なら malformed = 記録不可**）

反映先:
- 採用 → 次段へ
- 枯渇（全○にならず棄却） → **DLS に rejected + 理由**を書き戻す（rejected_hypothesis 形式）
- 未探索保留 → **dormant + 採らなかった理由**（rejected_alternatives）
- **dormant 候補を再活性化して採用する場合**（前候補を試行→棄却し別候補に乗り換え = **micro backtrack**）→
  決着エントリの `why` / `rejected_hypothesis` に「X を試行・棄却し dormant の Y を再活性化した」と明示し、
  **暗黙 backtrack を legible に残す**（DLS-168 痛み1。frontier 内ゆえ heavy な `/dls-backtrack` は不要 = cross-decision でない）。

### STEP 7: BACKTRACK
ノードが全滅したら上位 goal リーフへ戻る（**2 段まで**・同一 working tree 内）。
**3 段目 = goal リーフ全滅**（run 内で戻り切れない = cross-decision 領域）→ **`/dls-backtrack` を差し出す**
（主経路の内部 suggest、DLS-168 起動経路）。自動実行せず人間に「過去判断に戻るか」を確認してから handoff
（git 復元を伴うなら assist で人間承認）。さらなる discuss 差し戻しは Refinement。

## やらないこと（Refinement / 重い版 — spec §6）
- 明示 3 階層 tree（flat frontier で足りる、フォーク1=B）
- BO / 期待値エンジン（既存 `run_experiments_bayesian` との境界含む）
- 学習基準の再利用 / fidelity ladder 明示 / 意味検索 dedup / 3 段目 backtrack
- tier2/tier3 の LLM 分類ルータ（フォーク2 案 F1）
- dls-tree（薄い projection builder）は別増分。小タスクは「1 ノード tree」で本 skill 単独でも回る
