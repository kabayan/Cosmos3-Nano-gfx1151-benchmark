# Todo

`tasks/todo.md` は **active なタスク + マスターへのリンク情報** のみ保持する。
完了した項目は `/dls-commit` の Phase 1.7 で削除される（永続化先: コミット履歴 / DLS / `lessons.md`）。
運用ルールの詳細は `.claude/.dls/skills/06_todo_hygiene.md` を参照。

DLS-123: タスク本体は本ファイルに一元化する。`tasks/next.md` の「次のアクション」セクションは
本ファイルへの参照リンクのみで、タスク文言を直接書かない（DRY）。

---

## Active

- [ ] golden MSE の最終決着: **CUDA 参照 run**（E4 が仮説を棄却したため優先度が戻った）。公開コード b3967db + 同一入力（`inputs/omni/action_policy_robot.json`、seed 0）を CUDA GPU（AWS g6e.xlarge L40S 等）で 1 本実行し golden MSE を採点する。PASS(≈0.013) なら本環境固有の問題が残存 / FAIL(≈0.13) なら公開資産では記事値が再現しないことになり ROCm 移植は無罪。**環境調達を要するためユーザー判断待ち**。前提知見: (1) E1〜E3 fp32 感度実験で数値精度仮説は全棄却（DLS-012）、(2) E4 で checkpoint 版差仮説も棄却（DLS-014）、(3) 公開 v2 の policy 精度は NVIDIA 公式にも第三者にも測定記録が無い（公式 action ベンチは ID/FD のみで policy 指標なし、`golden_mse_max` を読むコードは framework に存在せず CI も numeric golden 対象外）
- [ ] （次の Policy run で確認）カーネルキャッシュ持ち越し（DLS-015）有効下での golden MSE が既存帯 0.126〜0.134 に入ることの確認、および Policy 本体でのコールドスタート短縮幅の実測（プローブでは全体 −14% / 大 conv 初回 −21%、本体は未測定）
- [ ] guidance 不一致（DLS-010）の実測決着: T2V を `--guidance 6.0` で 1 run し transformer_forward が約 2 倍になるかを確認（CFG 2 回順伝播の実証）。あわせて公式デフォルト条件での実測値を取得し README 倍率の扱いを判断する。**GPU 実行はユーザー承認待ち**
- [ ] README §2 の対外表現の訂正判断: 「すべての処理で…同一条件で実行」は (1) guidance 不一致の可能性（DLS-010）、(2) golden 品質基準の不合格（DLS-011）と両立しない。訂正の方向（条件差の注記 / 公式デフォルトでの再測定 / golden 不合格の帰属注記）は CUDA 参照 run と guidance 検証の結果を待って決める。「対論文比」列見出しと `docs/cosmos3_rocm_optimization_analysis.md` L17-19 の「論文値」表記は基準がすべて記事値のため独立に修正可能（DLS-006 確定事項）
- [ ] TeaCache 品質差の定量評価（DLS-004）: **実装済み（実験ブランチ `experiment/teacache-quality-eval` の `3d3b514`）、9 run 中 2 run 完了**。完了済: `baseline_run1` / `thresh_0.00`（恒等性検証は合格 — アクション 160 要素が完全一致、vision.mp4 の md5 も一致、forward_calls=30）。残り 7 run = `sanity_10.0` → `calib_logonly`（rel_l1 分布で閾値ラダー補正）→ `baseline_run2` + `thresh_{0.03,0.06,0.10,0.15}` → `scripts/compare_teacache_quality.py` で全ペア比較 → `docs/cosmos3_teacache_quality_eval_YYYYMMDD.md` 作成（**速度数値は一切記載しない**、DLS-003 制約）。実行: `python scripts/run_teacache_quality_matrix.py <run名>`。前提: `/tmp/cosmos-framework` が rsync 済（消えていたら `rsync -a --exclude .git temp_src/ /tmp/cosmos-framework/`）
- [ ] （任意・小幅改善系）torch 2.13.0 (rocm7.2 wheel) 更新検証。**ROCm #5750（Strix Halo 低クロック張り付き）は非該当を確認済**（amd-smi 実測で SCLK 2899MHz / MCLK 1000MHz = 3 段構成の最上位）。参照: docs/cosmos3_rocm_further_speedup_reassessment_20260726.md §5-①②
- [ ] （任意）DLS-005 の assumption（「41.66 → 42.88 秒の +2.9% は run-to-run 変動の範囲」、confidence: medium、ノイズフロア未測定）の再評価: 今セッションで同一フラグの反復（v4 vs v6、`--policy-sync-profile` のみ）が取れ、`generate_batch` の変動は **+0.79%**（124.906 → 125.898 秒）だった。ただし headline は `--policy-condition-cache` 付きで測定条件が異なるため直接比較できない。同フラグでの反復を 1 本取れば +2.9% がフロア内かを判定できる。フロア外なら DLS-005 の「環境劣化ではない」判断に留保が付く

## Reference Links

（参照リンクなし）
