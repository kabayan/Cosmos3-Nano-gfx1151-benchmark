# 原本: 対外比較の基準値と conditioning 除外の監査

- 日付: 2026-07-26
- 種別: doc（監査ログ）
- 文脈: DLS-005（41.66 秒の再現手順確定）の派生論点。ユーザー指示「論点を先に片付ける」により、
  conditioning 除外の妥当性と基準値の出所を調査した。

---

## 論点 1: 41.66 秒は conditioning を含まない

`--policy-condition-cache`（script L234 / L258 / L309-318）は measured フェーズの conditioning を
warmup1 で計算した結果で置換する。読み出しは measured フェーズのみ、書き込みは warmup フェーズの
初回のみ（warmup2 は再計算する）。

v3 run（`result/mainline_repro_v3_20260726/`）実測:

| フェーズ | generate_batch | generate_samples | decode |
|---|---|---|---|
| warmup1 | 2605.12 | 916.33 | 1688.78 |
| warmup2（conditioning 計算あり） | 125.85 | 116.70 | 7.4985 |
| measured（conditioning キャッシュ） | 42.88 | 35.25 | 0.0053 |

warmup2 と measured の generate_samples 差 **81.45 秒**が conditioning 相当。
generate_batch ベースでは 125.85 − 42.88 = **82.97 秒**。

最終報告の記述「すべての同期処理時間の総和」「見かけ上の高速化となる非同期処理を行わず」は、
実際には同期処理 約 83 秒がキャッシュにより測定対象外である点で過大である。

## 論点 2: 「論文値 29.00 秒」に出典が無い

調査で判明した経緯:

1. 記事本文（https://dev.classmethod.jp/articles/dgx-spark-cosmos3-omni-world-model-policy/ 、
   2026-07-26 取得）の記載は
   「DGX Spark での実行は、640×480 × 17 フレームの予測動画と 16 ステップ × 10 次元の action を、
   **モデル常駐後 21 秒で出力**しました。」のみ。**内訳（前処理 / サンプリング / デコード）の記載は無い**。
2. 同じ 6/14 コミット `9e39392` 内の `docs/cosmos3_rocm_optimization_analysis.md` L19 は
   「Policy: 約 1965 秒（**論文値 21 秒**に対し 93.6 倍遅い）」と、記事の 21 秒をそのまま基準にしている。
3. 一方 `docs/cosmos3_rocm_policy_optimization_final_report.md` と README.md（`b37cfb7`）は
   「論文値 29.00 秒」＝ サンプリング 21.00 + デコード 8.00 と分解した値を基準にしている。
4. docs 全体を検索しても Policy 実行時間に関する arXiv / 技術報告への出典は存在しない
   （vLLM-Omni 論文の参照はあるが別件）。
5. ユーザー確認（2026-07-26）: 出所は上記記事であり、**21 秒（総時間）が唯一の一次情報**。
   29.00 秒に別出典は無い。

したがって「論文値 29.00 秒」は、記事の 21 秒（総所要時間）を**サンプリング単体と解釈し直し**、
出所不明のデコード 8.00 秒を加算して構成された値である。

影響: 基準が 21 → 29 秒に膨らむと 1.5 倍目標のラインも 31.50 → 43.50 秒に緩む。
41.66 秒は緩和後のラインは通るが、記事の 21 秒基準では通らない。

## 比較の帰結

| 比較の取り方 | 倍率 | 1.5 倍目標 |
|---|---|---|
| 現行主張: 41.66s ÷ 29.00s | 1.44x | 達成 |
| 41.66s ÷ 21s（記事の総時間、conditioning 除外のまま） | 1.98x | 未達 |
| conditioning 込み ÷ 21s | 測定中（v4 run） | 未達見込み |

v3 warmup2 の 125.85 秒を暫定値とすると約 6.0x。確定値は
`result/mainline_full_v4_20260726/`（`--policy-sync-profile`、condition-cache 無し、
warmup 2 + 測定）で取得する。

## 未決（ユーザー判断事項）

README.md / final_report.md の対外主張をどう訂正するか。少なくとも以下 2 点は事実として確定:

- 41.66 秒は conditioning 約 83 秒を含まない
- 基準値 29.00 秒に一次出典は無く、一次情報は記事の 21 秒（総時間）
