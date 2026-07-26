# Todo

`tasks/todo.md` は **active なタスク + マスターへのリンク情報** のみ保持する。
完了した項目は `/dls-commit` の Phase 1.7 で削除される（永続化先: コミット履歴 / DLS / `lessons.md`）。
運用ルールの詳細は `.claude/.dls/skills/06_todo_hygiene.md` を参照。

DLS-123: タスク本体は本ファイルに一元化する。`tasks/next.md` の「次のアクション」セクションは
本ファイルへの参照リンクのみで、タスク文言を直接書かない（DRY）。

---

## Active

- [ ] TeaCache 品質差の定量評価（DLS-003）: TeaCache4Cosmos を Policy 推論に一時適用し、同一 seed で有効/無効の生成物を比較（アクションテンソル誤差、動画の PSNR/LPIPS 等）。閾値 `rel_l1_thresh` は保守的な値から数点。**速度成果は本線に含めない**（比較レポートは品質差のみ）。参照: docs/cosmos3_rocm_further_speedup_reassessment_20260726.md §4.2・§6
- [ ] （任意・小幅改善系）torch 2.13.0 (rocm7.2 wheel) 更新検証 + Strix Halo 低クロック張り付き（ROCm #5750）該当確認。参照: 同レポート §5-①②

## Reference Links

（参照リンクなし）
