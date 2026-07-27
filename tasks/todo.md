# Todo

`tasks/todo.md` は **active なタスク + マスターへのリンク情報** のみ保持する。
完了した項目は `/dls-commit` の Phase 1.7 で削除される（永続化先: コミット履歴 / DLS / `lessons.md`）。
運用ルールの詳細は `.claude/.dls/skills/06_todo_hygiene.md` を参照。

DLS-123: タスク本体は本ファイルに一元化する。`tasks/next.md` の「次のアクション」セクションは
本ファイルへの参照リンクのみで、タスク文言を直接書かない（DRY）。

---

## Active

- [ ] TeaCache 品質差の定量評価（DLS-004）: **実装済み（実験ブランチ `experiment/teacache-quality-eval` の `3d3b514`）、9 run 中 2 run 完了**。完了済: `baseline_run1` / `thresh_0.00`（恒等性検証は合格 — アクション 160 要素が完全一致、vision.mp4 の md5 も一致、forward_calls=30）。残り 7 run = `sanity_10.0` → `calib_logonly`（rel_l1 分布で閾値ラダー補正）→ `baseline_run2` + `thresh_{0.03,0.06,0.10,0.15}` → `scripts/compare_teacache_quality.py` で全ペア比較 → `docs/cosmos3_teacache_quality_eval_YYYYMMDD.md` 作成（**速度数値は一切記載しない**、DLS-003 制約）。実行: `python scripts/run_teacache_quality_matrix.py <run名>`。前提: `/tmp/cosmos-framework` が rsync 済（消えていたら `rsync -a --exclude .git temp_src/ /tmp/cosmos-framework/`）
- [ ] （任意・小幅改善系）torch 2.13.0 (rocm7.2 wheel) 更新検証。**ROCm #5750（Strix Halo 低クロック張り付き）は非該当を確認済**（amd-smi 実測で SCLK 2899MHz / MCLK 1000MHz = 3 段構成の最上位）。参照: docs/cosmos3_rocm_further_speedup_reassessment_20260726.md §5-①②
- [ ] （任意）DLS-005 の assumption（「41.66 → 42.88 秒の +2.9% は run-to-run 変動の範囲」、confidence: medium、ノイズフロア未測定）の再評価: 今セッションで同一フラグの反復（v4 vs v6、`--policy-sync-profile` のみ）が取れ、`generate_batch` の変動は **+0.79%**（124.906 → 125.898 秒）だった。ただし headline は `--policy-condition-cache` 付きで測定条件が異なるため直接比較できない。同フラグでの反復を 1 本取れば +2.9% がフロア内かを判定できる。フロア外なら DLS-005 の「環境劣化ではない」判断に留保が付く
- [ ] （任意）MIOpen カーネルキャッシュのホストマウント: コンテナ使い捨てのため毎 run カーネル探索が走り、コールドスタートが 6 月比 7 倍（106 秒 → 730 秒）に伸びている。実験の所要時間短縮に効く見込み

## Reference Links

（参照リンクなし）
