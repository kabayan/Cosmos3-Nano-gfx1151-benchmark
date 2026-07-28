# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28
> 前セッションの要約: DLS-020 の限定検証を進め、公式 guidance T2V の厳密 und branch cache を採用（DLS-021）、PyTorch 2.13 + AOTriton 0.12b を出力非等価で不採用（DLS-022）とした。結果を README に反映し `1f2ee59` でコミットした。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。**

- HEAD: `1f2ee59`（DLS-021 / DLS-022、README・検証記録・隔離 Dockerfile）
- main は origin/main より先行した未 push コミットを保持
- DLS-020 の即時実行可能な限定検証は完了。残件は AOTriton PR #203/#205 の
  merge・Level 1 correctness・性能値公開を待ってから行う gfx1151/head_dim=128 probe のみ
- 現行採用 stack は PyTorch 2.9.1 + 検証済み TunableOp 表
- `experiment/teacache-quality-eval` は未マージ（DLS-004 の品質評価トラック）
- `.agents/`、`.codex/`、`AGENTS.md`、`agents/` は本作業前から存在する未追跡物。
  今回のコミットには含めず、そのまま保持した

### 公式 guidance の厳密 und branch cache

- T2I: 115.589 → 49.633 秒（-57.1%、2.33x speedup）、JPG byte 一致
- T2V: 55.561 → 40.806 秒（-26.6%、1.36x speedup）、MP4 byte 一致
- I2V: 192.521 → 45.622 秒（-76.3%、4.22x speedup）、MP4 byte 一致
- 各モードとも warmup + measured の計 140 transformer calls は
  2 writes / 138 reads / 0 invalidations。近似・計算省略ではない
- README の公式 guidance 値と cache 効果表を上記結果へ更新済み

### PyTorch 2.13 隔離比較

- `docker/cosmos3-rocm72-diffusers-torch213.Dockerfile` で再現可能な派生 image を作成
- image: `cosmos3-rocm72-diffusers:torch213`
  (`sha256:48ff721a2c79bfd6904705f1d0ee0e4b09441dbd0b05a9cf54ae3bc68b73a2d7`)
- torch 2.13.0 + torchvision 0.28.0 + Triton ROCm 3.7.1 + AOTriton 0.12.0 の GPU smoke は合格
- ROCm 7.2 index に torchaudio 2.13 wheel が無く、旧 2.9 binary は ABI 不整合のため派生 image から削除
- 旧 TunableOp 表は ROCBLAS_VERSION validator 不一致で安全に流用不可
- T2V は 53.361 秒（未調律）で、現行 40.806 秒より遅い。ただし性能の最終結論には使わない
- MP4 hash が変わり全 21 decoded frame が相違（PSNR 28.41 dB / SSIM 0.9488）したため、
  事前登録済みの hash 一致条件により不採用。2.13 用再調律には拡張しない

## 完了済み（今セッション）

- T2V 公式 guidance 条件で既存の厳密 und branch cache を実測し、26.6%短縮と byte 完全一致を確認
- DLS-021 と検証原本
  `.claude/.dls/raw/20260728_doc_t2v_und_branch_cache_official_guidance_verification.md` を追加
- PyTorch 2.13 + AOTriton 0.12b の隔離 image を構築・smoke・同一プロトコル比較
- 出力非等価を確認し、現行 PyTorch 2.9.1 stack 維持を DLS-022 に記録
- `README.md` に3モードの cache 効果と最新公式 guidance 値を反映
- 再評価レポートと `tasks/todo.md` を DLS-022 の結論に同期
- 上記7ファイルを `1f2ee59` でコミット

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体はtodo.mdに一元化）

## ブロッカー・注意事項

- AOTriton PR #203/#205 は未マージ・検証中。Draft版を本線へ手動導入しない
- diffusers 経路の非劣化条件は同一入力・seed の出力 hash 一致
- PyTorch 2.13 の未調律速度を 2.13 本体の性能結論として引用しない
- 2.9.1 の TunableOp 表を 2.13 へ validator 書き換えで流用しない
- benchmark 成果物は gitignore 対象の `result/` にあるため、再生成に必要な条件・hash・数値は raw に転記済み
- todo hygiene 候補: `tasks/todo.md` の完了済み torch 2.13 項目。ユーザー承認なしに削除しない
- CUDA参照 run は当分実施不可（既存のユーザー決定）
- third_party/diffusers と実行 image `/opt/diffusers` を乖離させない

## 関連ファイル

- `.claude/.dls/active.md`（DLS-020〜DLS-022）
- `.claude/.dls/raw/20260728_doc_t2v_und_branch_cache_official_guidance_verification.md`
- `.claude/.dls/raw/20260728_doc_pytorch_213_rocm72_isolated_t2v_verification.md`
- `README.md`（公式 guidance と厳密 cache 効果表）
- `docker/cosmos3-rocm72-diffusers-torch213.Dockerfile`
- `docs/cosmos3_rocm_further_speedup_reassessment_20260726.md`
- `tasks/todo.md`
- `result/t2v_und_cache_official_20260728/`
- `result/t2v_und_cache_torch213_official_20260728/`
