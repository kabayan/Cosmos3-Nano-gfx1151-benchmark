# GB10 / Strix Halo 外部 GEMM 比較調査

- 日付: 2026-07-28
- 種別: doc / web research
- 依頼: ユーザーから、README の固定 16384 角 GEMM 比較について、同様の公開ベンチを検索し、結果を README とは別ドキュメントにしてリンクするよう依頼された

## ローカル原本

- `scripts/bench_peak.py`: 16384 × 16384 の `torch.mm` を同期測定
- `README.md` §4: GB10 記事値と Radeon 8060S / gfx1151 実測の比較
- `result/cfg_batch_probe/gemm_bf16.json`: Cosmos3 実形状で BF16 最大 36.1105 TFLOPS
- `docs/dgx_spark_comparison_report.md`: 既存の固定形状比較レポート

固定 16384 角の BF16 は GB10 96.91 TFLOPS、本環境 20.91 TFLOPS、比率 4.63 倍。

## 外部調査結果

### 直接比較として最も強い資料

The Register は DGX Spark と Strix Halo（HP Z2 Mini G1a）を同一記事内で直接測定した。
Max Achievable MatMul FLOPS（MAMF）の BF16 結果は次の通り。

- GB10: 101 TFLOPS
- Strix Halo: 46 TFLOPS
- 比率: 2.20 倍

URL:
https://www.theregister.com/on-prem/2025/12/25/tested_amds_strix_halo_vs_nvidias_dgx_spark/2098514

MAMF は多数の M/N/K 形状から最高到達値を探す測定であり、固定 16384 角とは目的が異なる。

### Strix Halo の形状走査

stas00 の MAMF スクリプトを Strix Halo で約2日間走らせたコミュニティ結果があり、BF16 で 30 TFLOPS 超の形状が報告された。

- 投稿: https://www.reddit.com/r/ROCm/comments/1ocxxw6/exploring_strix_halo_bf16_tflops_my_2day/
- MAMF 実装: https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks
- 可視化: https://johnnytshi.github.io/strix_halo_bf16_tflops/

### 8192 角のコミュニティ集計

BF16 8192 × 8192 の PyTorch 行列積集計では DGX Spark 約 60 TFLOPS、Strix Halo 約 36 TFLOPSと記載されている。ただし投稿者自身が両値を未確認のオンライン値として扱っており、正式な根拠には使わない。

URL:
https://www.reddit.com/r/LocalLLaMA/comments/1pkbmqe/tflops_by_gpu/

### 実アプリ比較

The Register の同じ直接比較では、単一 batch LLM decode は両機が近く、compute-bound な prompt processing は GB10 が約 2〜3 倍、FLUX.1 Dev は約 2.5 倍と報告された。

Signal65 の比較では BF16 画像生成で GB10 が約 1.3〜2 倍、Wan 2.2 14B 動画生成で 7.6 倍だった。ただし NVIDIA 関連の調査であり、AMD 側で FP16 / FP8 workload が実行不能だったなどソフトウェア成熟度も含むため、純粋な GPU GEMM 比較には使わない。

URL:
https://signal65.com/wp-content/uploads/2026/03/Signal65-Insights_NVIDIA-DGX-Spark-Platform-Arm-and-NVIDIA-Reinvent-the-Workstation.pdf

## 解釈

- 固定 16384 角の 4.63 倍は、その単一形状と各 software stack を含む実測差として有効
- MAMF の 2.20 倍は、各 GPU が得意な形状で到達できる最大 BF16 GEMM の差
- GB10 は固定形状 96.91 TFLOPS が外部 MAMF 101 TFLOPS の約 96%に達する
- 本環境は固定形状 20.91 TFLOPSだが、Cosmos3 実形状で 36.11 TFLOPS、外部 Strix Halo MAMF で 46 TFLOPSが観測される
- したがって、4.63 倍を一般的な BF16 GPU 能力差へ拡張すると過大。形状・kernel・runtime が差を増幅している
- CUDA の有無だけを原因にできない。ROCm 側も WMMA / hipBLASLt 系 GPU kernel を使用しており、ハードウェア規模と precision ごとの演算器構成、kernel 選択、software stack の総合差である

## ドキュメント構成判断

- 採用: 既存 `docs/dgx_spark_comparison_report.md` を外部検証込みの詳細レポートへ拡張し、README からリンクする
- 不採用: 外部比較だけの新規レポートを追加する。固定形状レポートと結論・数値が重複し、参照先が分散する
- 不採用: README の固定形状表を MAMF 値へ置換する。測定目的が異なり、ローカルの再現可能な実測を失う
