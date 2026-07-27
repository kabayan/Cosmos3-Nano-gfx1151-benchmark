# Todo

`tasks/todo.md` は **active なタスク + マスターへのリンク情報** のみ保持する。
完了した項目は `/dls-commit` の Phase 1.7 で削除される（永続化先: コミット履歴 / DLS / `lessons.md`）。
運用ルールの詳細は `.claude/.dls/skills/06_todo_hygiene.md` を参照。

DLS-123: タスク本体は本ファイルに一元化する。`tasks/next.md` の「次のアクション」セクションは
本ファイルへの参照リンクのみで、タスク文言を直接書かない（DRY）。

---

## Active

- [ ] golden MSE 不合格（DLS-011/012/013）の決着 **E4: v1 重み ROCm golden 照合**。HF `nvidia/Cosmos3-Nano` branch `spectralflight/shim`（revision `35c5cd345`、約 30GB）= v1_midtrain iter12000 EMA 重みを取得し、tensor 名リネーム対応（旧 814 key → 新形式、key 数は同一 1165 で機械導出可能な見込み）+ config 整合の移植をしたうえで本環境（ROCm）で golden 照合 1 run。**PASS(≈0.013) なら ROCm 完全無罪 + v1→v2 checkpoint 差し替えが原因と確定（CUDA run 不要）**。FAIL はロード誤りと識別不能（PASS のみが情報を持つ非対称リスク）。撤退ライン: リネーム移植の工数が想定超過したら中断し CUDA 参照 run に切替。**30GB DL + GPU 実行のためユーザー承認待ち**。原本: `.claude/.dls/raw/20260727_doc_tokenizer_pin_forensics_and_v1v2_checkpoint_swap.md`
- [ ] （E4 が FAIL または移植不能だった場合の後続）golden MSE の最終決着: **CUDA 参照 run**。公開コード b3967db + 同一入力（`inputs/omni/action_policy_robot.json`、seed 0）を CUDA GPU（AWS g6e.xlarge L40S 等）で 1 本実行し golden MSE を採点する。PASS(≈0.013) なら ROCm 側に精度以外の意味論差が残存 / FAIL(≈0.13) なら上流版差で確定し ROCm 移植は完全無罪。**環境調達を要するためユーザー判断待ち**。前提知見: E1〜E3 fp32 感度実験で数値精度仮説は全棄却済み（DLS-012、原本 `.claude/.dls/raw/20260727_doc_policy_golden_mse_precision_sweep.md`）
- [ ] guidance 不一致（DLS-010）の実測決着: T2V を `--guidance 6.0` で 1 run し transformer_forward が約 2 倍になるかを確認（CFG 2 回順伝播の実証）。あわせて公式デフォルト条件での実測値を取得し README 倍率の扱いを判断する。**GPU 実行はユーザー承認待ち**
- [ ] README §2 の対外表現の訂正判断: 「すべての処理で…同一条件で実行」は (1) guidance 不一致の可能性（DLS-010）、(2) golden 品質基準の不合格（DLS-011）と両立しない。訂正の方向（条件差の注記 / 公式デフォルトでの再測定 / golden 不合格の帰属注記）は CUDA 参照 run と guidance 検証の結果を待って決める。「対論文比」列見出しと `docs/cosmos3_rocm_optimization_analysis.md` L17-19 の「論文値」表記は基準がすべて記事値のため独立に修正可能（DLS-006 確定事項）
- [ ] TeaCache 品質差の定量評価（DLS-004）: **実装済み（実験ブランチ `experiment/teacache-quality-eval` の `3d3b514`）、9 run 中 2 run 完了**。完了済: `baseline_run1` / `thresh_0.00`（恒等性検証は合格 — アクション 160 要素が完全一致、vision.mp4 の md5 も一致、forward_calls=30）。残り 7 run = `sanity_10.0` → `calib_logonly`（rel_l1 分布で閾値ラダー補正）→ `baseline_run2` + `thresh_{0.03,0.06,0.10,0.15}` → `scripts/compare_teacache_quality.py` で全ペア比較 → `docs/cosmos3_teacache_quality_eval_YYYYMMDD.md` 作成（**速度数値は一切記載しない**、DLS-003 制約）。実行: `python scripts/run_teacache_quality_matrix.py <run名>`。前提: `/tmp/cosmos-framework` が rsync 済（消えていたら `rsync -a --exclude .git temp_src/ /tmp/cosmos-framework/`）
- [ ] （任意・小幅改善系）torch 2.13.0 (rocm7.2 wheel) 更新検証。**ROCm #5750（Strix Halo 低クロック張り付き）は非該当を確認済**（amd-smi 実測で SCLK 2899MHz / MCLK 1000MHz = 3 段構成の最上位）。参照: docs/cosmos3_rocm_further_speedup_reassessment_20260726.md §5-①②
- [ ] （任意）DLS-005 の assumption（「41.66 → 42.88 秒の +2.9% は run-to-run 変動の範囲」、confidence: medium、ノイズフロア未測定）の再評価: 今セッションで同一フラグの反復（v4 vs v6、`--policy-sync-profile` のみ）が取れ、`generate_batch` の変動は **+0.79%**（124.906 → 125.898 秒）だった。ただし headline は `--policy-condition-cache` 付きで測定条件が異なるため直接比較できない。同フラグでの反復を 1 本取れば +2.9% がフロア内かを判定できる。フロア外なら DLS-005 の「環境劣化ではない」判断に留保が付く
- [ ] （任意）MIOpen カーネルキャッシュのホストマウント: コンテナ使い捨てのため毎 run カーネル探索が走り、コールドスタートが 6 月比 7 倍（106 秒 → 730 秒）に伸びている。実験の所要時間短縮に効く見込み

## Reference Links

（参照リンクなし）
