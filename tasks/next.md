# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28
> 前セッションの要約: GB10（DGX Spark）と Strix Halo（EVO-X2）の行列積差を、ローカル実測と外部 MAMF ベンチで再評価した。固定形状の 4.63x 差を CUDA の有無だけに帰属せず、GPU 規模・Tensor Core / WMMA 経路・ソフトウェアスタック・形状依存を含む差として整理し、DLS-023、専用比較レポート、README リンクを `fb3b4e1` でコミット・push 済み。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。**

- 作業開始時の HEAD: `fb3b4e1`（DLS-023、比較レポート・README・調査原本）
- `fb3b4e1` までは main と origin/main が同期済み
- 現行採用 stack は PyTorch 2.9.1 + 検証済み TunableOp 表
- DLS-020 の即時実行可能な限定検証は完了。残件は AOTriton PR #203/#205 の
  merge・Level 1 correctness・性能値公開を待ってから行う gfx1151/head_dim=128 probe のみ
- `experiment/teacache-quality-eval` は未マージ（DLS-004 の品質評価トラック）
- `.agents/`、`.codex/`、`AGENTS.md`、`agents/` は本作業前から存在する未追跡物。
  今回のコミットにも含めず、そのまま保持した

### GB10 / Strix Halo 行列積比較

- ローカル固定形状 `16384³` BF16 GEMM は GB10 96.91 TFLOPS、Strix Halo 20.91 TFLOPSで 4.63x
- 外部 MAMF BF16 は GB10 101 TFLOPS、Strix Halo 46 TFLOPSで 2.20x
- ローカル Cosmos3 実形状の最大値は Strix Halo 36.1105 TFLOPS
- 4.63x は同一固定形状と各ソフトウェア経路を含む実測差であり、GPU の一般的な最大性能差でも
  CUDA 単独の効果でもない
- GB10 はより大きい GPU と Tensor Core を持ち、Strix Halo は RDNA 3.5 の WMMA 経路を使う。
  ハードウェア規模、演算器、ライブラリ成熟度、kernel、形状の複合差として扱う
- 固定 `16384³` は再現性のある同一形状比較、MAMF は各 GPU の高い実効値探索で目的が異なる
- README §4 から専用レポート `docs/dgx_spark_comparison_report.md` へリンク済み

### 直前までの検証状態

- 公式 guidance の厳密 und branch cache は T2I 2.33x、T2V 1.36x、I2V 4.22xで、各出力は byte 一致（DLS-021）
- PyTorch 2.13 + AOTriton 0.12b は T2V 出力非等価のため不採用。現行 2.9.1 stack を維持（DLS-022）

## 完了済み（今セッション）

- CUDA の有無だけで GEMM 差を説明できるかを再検討
- GB10 と Strix Halo の GPU 規模、Tensor Core / WMMA 経路、ローカル実測を比較
- 同様の外部ベンチを検索し、The Register の MAMF BF16 結果を比較材料として追加
- `docs/dgx_spark_comparison_report.md` を専用の比較・解釈ドキュメントとして拡充
- `README.md` から専用レポートへリンクし、4.63x の解釈上の注意を追記
- 判断を DLS-023、調査根拠を
  `.claude/.dls/raw/20260728_doc_gb10_strix_halo_external_gemm_comparison.md` に記録
- 上記変更を `fb3b4e1` でコミットし、origin/main へ push

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体はtodo.mdに一元化）

## ブロッカー・注意事項

- 外部 MAMF は別筐体・別 runtime・探索形状の結果であり、この環境での再現測定ではない
- 同一 MAMF harness を GB10 と EVO-X2 の両方でローカル実行した比較は未実施
- 固定 `16384³` の 4.63x を、一般的な GPU 最大性能差または CUDA 単独効果へ一般化しない
- AOTriton PR #203/#205 は未マージ・検証中。Draft版を本線へ手動導入しない
- benchmark 成果物は gitignore 対象の `result/` にあるため、再生成に必要な条件・hash・数値は raw に転記済み
- todo hygiene 候補: `tasks/todo.md` の完了済み torch 2.13 項目。ユーザー承認なしに削除しない
- CUDA参照 run は当分実施不可（既存のユーザー決定）

## 関連ファイル

- `.claude/.dls/active.md`（DLS-020〜DLS-023）
- `.claude/.dls/raw/20260728_doc_gb10_strix_halo_external_gemm_comparison.md`
- `.claude/.dls/raw/20260728_doc_t2v_und_branch_cache_official_guidance_verification.md`
- `.claude/.dls/raw/20260728_doc_pytorch_213_rocm72_isolated_t2v_verification.md`
- `docs/dgx_spark_comparison_report.md`
- `docs/cosmos3_rocm_further_speedup_reassessment_20260726.md`
- `README.md`
- `tasks/todo.md`
