# DGX Spark (GB10) と Radeon 8060S (gfx1151) の GEMM 比較

## 1. 結論

固定 `16384 × 16384` の `torch.mm` では、DGX Spark (GB10) は本環境の Radeon 8060S / gfx1151 に対して FP16/BF16 で約 4.6〜4.8 倍高速だった。この値は**同じ単一形状における各 software stack 込みの実測差**として有効だが、GPUの一般的な最大行列積性能差を意味しない。

外部の直接比較では、形状を走査して最高到達値を探す MAMF（Max Achievable MatMul FLOPS）で BF16 が GB10 101 TFLOPS、Strix Halo 46 TFLOPS、差は約 2.2 倍だった。本環境でも Cosmos3 の実形状では最大 36.11 TFLOPSを観測している。

したがって、固定16384角で観測した4.63倍差には、ハードウェア規模だけでなく**行列形状、選択されたkernel、CUDA/cuBLASとROCm/hipBLASLtの実装差**が含まれる。「CUDAがあるから4.63倍」と単一要因へ帰属させることはできない。

## 2. 比較対象

| 項目 | DGX Spark | 本環境（EVO-X2） |
|---|---|---|
| SoC / GPU | NVIDIA GB10 Grace Blackwell | AMD Ryzen AI Max+ 395 / Radeon 8060S |
| GPU architecture | Blackwell | RDNA 3.5 (`gfx1151`) |
| 行列演算経路 | Tensor Core / CUDA / cuBLAS | WMMA系 / ROCm / rocBLAS・hipBLASLt |
| unified memory | 128GB LPDDR5X | 128GB LPDDR5X（GPU割当は環境設定依存） |
| memory interface | 256-bit | 256-bit |
| runtime | CUDA 12.x / PyTorch | ROCm 7.2 / PyTorch 2.9.1 |

PyTorch ROCm は互換APIとして `torch.cuda` 名前空間を使用するが、実体はNVIDIA CUDAではない。`scripts/bench_peak.py` の `torch.mm` は、各環境でそれぞれのGPU行列積libraryへdispatchされる。

## 3. 固定16384角の比較

GB10側はHPCシステムズの記事「[DGX Sparkで見る“デスクサイドAI”の実力](https://www.hpc.co.jp/tech-blog/2026/03/17/dgx-spark-desk-side/)」、gfx1151側は `scripts/bench_peak.py` の実測である。

| 精度 | GB10 | Radeon 8060S / gfx1151 | GB10 / gfx1151 |
|---|---:|---:|---:|
| FP16 | 96.54 TFLOPS | 20.04 TFLOPS | 4.82倍 |
| BF16 | 96.91 TFLOPS | 20.91 TFLOPS | 4.63倍 |
| FP32（TF32無効） | 18.54 TFLOPS | 2.86 TFLOPS | 6.48倍 |
| FP64 | 0.42 TFLOPS | 0.45 TFLOPS | 0.93倍 |

測定条件:

- 行列サイズ: `16384 × 16384`
- warmup: 3回
- measured: FP16/BF16 50回、FP32 10回、FP64 5回
- 演算: `torch.mm`
- 測定区間の前後でGPU同期
- FLOP数: `2 × M × N × K`

この結果から分かるのは、GB10がこの固定形状で高い利用率を達成していることと、Radeon側もFP16/BF16でFP32比約7倍のWMMA効果を得ていることである。FP64の逆転は、精度ごとに演算器配分が異なることを示しており、「CUDAの有無が全精度を一様に速くする」という説明とは整合しない。

## 4. 外部の類似GEMMベンチ

### 4.1 両機を直接測定したMAMF

[The Registerの直接比較](https://www.theregister.com/on-prem/2025/12/25/tested_amds_strix_halo_vs_nvidias_dgx_spark/2098514) は、DGX SparkとStrix Halo搭載HP Z2 Mini G1aを同一記事内で測定している。

| BF16 MAMF | 実測 | 対理論値（記事推定） |
|---|---:|---:|
| GB10 | 101 TFLOPS | 約81%（理論125 TFLOPS） |
| Strix Halo / Radeon 8060S | 46 TFLOPS | 約82%（理論約56 TFLOPS） |
| 比率 | 2.20倍 | ― |

MAMFは多数のM/N/K候補から最も速い形状を探す。固定16384角の再現性確認とは目的が異なるため、101対46でREADMEの固定形状表を置換してはいけない。一方で、一般的な「最大BF16 GEMM能力差」を論じる場合は、固定形状1点よりこちらが適切である。

### 4.2 Strix Haloの形状走査

コミュニティでは、公開されている[stas00のMAMFスクリプト](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks)をStrix Haloで約2日間実行した[BF16形状走査](https://www.reddit.com/r/ROCm/comments/1ocxxw6/exploring_strix_halo_bf16_tflops_my_2day/)があり、30 TFLOPSを超える形状が複数報告されている。

また、BF16 `8192 × 8192` の[コミュニティ集計](https://www.reddit.com/r/LocalLLaMA/comments/1pkbmqe/tflops_by_gpu/)にはDGX Spark約60 TFLOPS、Strix Halo約36 TFLOPSとある。ただし投稿者自身がこの2値を未確認のオンライン値として扱っているため、傾向確認にだけ用い、正式な比較値には採用しない。

### 4.3 Cosmos3実形状

`result/cfg_batch_probe/gemm_bf16.json` では、Cosmos3 transformer由来の実形状で最大36.1105 TFLOPSを観測した。例えば `N=3808, K=4096, M=12288` のFFN projectionであり、固定16384角の20.91 TFLOPSを大きく上回る。

これはRadeon 8060Sの上限が20.91 TFLOPSではなく、形状とkernel選択によって少なくとも36 TFLOPS級まで変化することをローカルでも示している。

## 5. 数値が違う理由

| 比較 | GB10 / Strix Halo | 意味 |
|---|---:|---|
| 固定16384角 | 4.63倍 | 指定した単一形状＋各runtimeの差 |
| MAMF最高値 | 2.20倍 | 各GPUが得意な形状での最大BF16 GEMM差 |
| 8192角コミュニティ値 | 約1.67倍 | 未検証の参考値 |

固定16384角では、GB10の96.91 TFLOPSは外部MAMF 101 TFLOPSの約96%に達する。一方、本環境の20.91 TFLOPSは外部Strix Halo MAMF 46 TFLOPSの約45%で、ローカル実形状最大36.11 TFLOPSよりも低い。

ここから、4.63倍のうち相当部分はRadeon側の固定形状に対するkernel適合度やlibrary選択によって拡大している、と推定できる。ただし、外部MAMFは別筐体・別runtimeであり、差の全量を特定のlibraryへ割り当てることはできない。

## 6. 実アプリでの差

同じThe Register比較では、単一batchのLLM decodeは両機が近い一方、compute-boundになりやすいprompt processingではGB10が約2〜3倍、FLUX.1 Dev画像生成では約2.5倍だった。decodeはメモリ帯域、prompt processingと画像生成はGEMM演算性能の影響が強いため、MAMFの約2.2倍差と方向が整合する。

[Signal65の比較レポート](https://signal65.com/wp-content/uploads/2026/03/Signal65-Insights_NVIDIA-DGX-Spark-Platform-Arm-and-NVIDIA-Reinvent-the-Workstation.pdf)では、BF16画像生成でGB10が約1.3〜2倍、Wan 2.2 14B動画生成で7.6倍だった。ただし、この資料はNVIDIA関連の調査で、AMD側が一部FP16/FP8 workloadを実行できないなどsoftware対応状況も含む。純粋なGEMMハードウェア差ではなく、end-to-end stack差の参考として扱う。

## 7. 適切な読み方

- **固定形状の再現比較**には `96.91 vs 20.91 TFLOPS（4.63倍）` を使う
- **最大BF16 GEMM能力の比較**には外部MAMFの `101 vs 46 TFLOPS（2.20倍）` を参考にする
- **Cosmos3の性能上限**には実形状の36.11 TFLOPSとend-to-end測定を使い、固定16384角を上限にしない
- **CUDAの効果**を論じる際は、GPU規模、演算器構成、kernel選択、runtime成熟度と分離不能であることを明記する
- NVIDIAの1 PFLOP FP4（sparsity込み）とAMDの50 NPU TOPSはprecision・sparsity・実行デバイスが異なるため、このBF16 GPU GEMM表へ混ぜない

## 8. 再現方法

本環境の固定形状測定:

```bash
python3 scripts/bench_peak.py
```

最大到達形状を比較する場合は、固定形状スクリプトとは別にMAMFの同一revision・同一探索範囲を両機で実行する必要がある。現時点では外部記事値の参照であり、本プロジェクトが両機でMAMFを再測定した結果ではない。
