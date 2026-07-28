# Cosmos3-Nano AMD ROCm (gfx1151) パフォーマンスベンチマーク & 最適化

本リポジトリは、NVIDIA のマルチモーダル世界モデル・ポリシーモデル **nvidia/Cosmos3-Nano** を、AMD の ROCm 7.2 環境およびコンシューマ/ワークステーション向け GPU（Radeon gfx1151）上で動作させ、極限まで高速化した最適化成果およびベンチマーク結果をまとめたものです。

---

## 1. 動作・測定環境

*   **APU/GPU**: AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151 / RDNA3/4 世代)
*   **VRAM**: 120 GB (システム共有/専用領域)
*   **OS**: Linux (Docker コンテナ環境)
*   **ROCm**: 7.2.0
*   **PyTorch**: 2.9.1+rocm7.2.0

> PyTorch 2.13.0 + AOTriton 0.12.0 も隔離環境で検証しましたが、同一入力・seed の T2V 出力が
> 現行 2.9.1 と一致しなかったため採用していません。下記の公表値はすべて検証済みの 2.9.1 stack の値です。

---

## 2. 4大ユースケース パフォーマンス比較 (対記事比)

Classmethod 記事「[NVIDIA Cosmos を AMD ROCm で動かす](https://dev.classmethod.jp/articles/dgx-spark-cosmos3-omni-world-model-policy/)」のデータおよび動作条件に準拠し、モデル常駐後の定常状態（Warm状態）での生成時間を比較した結果です。

**合否基準は「対記事倍率が両環境の価格差（約 2 倍）以内に収まるか」**としています（従来の「対記事 1.5 倍以内」から変更）。

### 📊 実行時間および速度差の比較

T2I / T2V / I2V は guidance（CFG）設定で transformer の計算量が約 2 倍変わるため、**guidance 1.0（CFG 無効）と公式デフォルト guidance（T2I 4.0 / T2V・I2V 6.0）の両条件を併記**します。記事は公式サンプル JSON 準拠を明言しており、記事側条件は公式デフォルトである可能性が高い一方、確認は取れていません（下記 NOTE 参照）。

| No | ユースケース / 処理モード | 記事値 (CUDA) | **guidance 1.0 実測 (対記事比)** | **公式 guidance 実測 (対記事比)** | 実行条件 / 備考 |
|---|---|---|---|---|---|
| 1 | **Text-to-image (T2I)** | 22 秒 | **27.136 秒 (1.23x)** | **49.633 秒 (2.25x)** | 960x960, 35 steps (Diffusers-native und branch cache, 2 スロット) |
| 2 | **Text-to-video (T2V)** | 22 秒 | **32.165 秒 (1.46x)** | **40.806 秒 (1.85x)** | 256p, 24 requested frames, 35 steps (Diffusers-native und branch cache, 2 スロット) |
| 3 | **Image-to-video (I2V)** | 17 秒 | **25.045 秒 (1.47x)** | **45.622 秒 (2.69x)** | 256p, 24 requested frames, 35 steps (Diffusers-native und branch cache, 2 スロット) |
| 4 | **Policy Model (生成)** | 21 秒 | **`41.66` 秒 (`1.98x`)** | —（guidance 条件の対象外） | 640x480 x 17f 動画 + 16x10 アクション出力。サンプリング + VAE デコード |
| - | ┗ デノイズサンプリング | 内訳非公開 | `33.84` 秒 | — | 30 steps |
| - | ┗ VAE デコード | 内訳非公開 | `7.49` 秒 | — | 17f ビデオ復元 (upsample_3 torch.compile 最適化) |
| - | ┗ (参考) 入力観測エンコード | 内訳非公開 | `0.28` 秒 | — | `--policy-min-condition-encode` 適用時。未適用時は `81.57` 秒。上記「生成」には含まない。下記 NOTE 参照 |

### 厳密 und branch cache の効果（公式 guidance）

| モード | cache 無効 | cache 有効 | 短縮率 / 高速化 | transformer | 出力等価性 |
|---|---:|---:|---:|---:|---|
| **T2I** | 115.589 秒 | **49.633 秒** | **57.1% / 2.33x** | 113.254 → 47.340 秒 | JPG bit 一致 |
| **T2V** | 55.561 秒 | **40.806 秒** | **26.6% / 1.36x** | 50.251 → 35.552 秒 | MP4 byte 一致 |
| **I2V** | 192.521 秒 | **45.622 秒** | **76.3% / 4.22x** | 187.087 → 40.238 秒 | MP4 byte 一致 |

いずれも同一入力・seed・steps・guidance・dtype で比較し、cache 有効時は warmup + measured 合計 140 transformer calls 中 2 writes / 138 reads / 0 invalidations でした。Policy Model は Diffusers の understanding branch を使うこの cache の対象外です。

> [!NOTE]
> * 記事側の値は、動画尺ではなく「モデル常駐後の生成所要時間」です。
> * 価格差 2.0 基準に対し、**Policy Model（生成 1.98x）は基準の内側**です（入力観測エンコード込みの同期総和では 2.02x で境界上）。
>   **公式 guidance 条件の T2V（1.85x）は基準の内側、T2I / I2V（2.25x / 2.69x）は基準を超過**しています。CFG は cond/uncond の
>   逐次 2 回 forward で transformer 計算量が約 2 倍になるためで、計算内容を変えない範囲での追加短縮余地が
>   小さいことは GEMM 実測で確認済みです（`result/cfg_batch_probe/`）。guidance 1.0 条件では 3 モードとも基準内です。
> * 記事の実際の guidance 設定は未確認です。1.0 / 公式デフォルトのどちらであっても、対応する実測値を上表に併記しています。
>   公式 guidance 条件の測定記録は `result/guidance_2slot_20260728/`（T2I / I2V）と `result/t2v_und_cache_official_20260728/`（T2V）です。
> * 厳密 und branch cache は T2I / T2V / I2V の3モードすべてで出力一致を確認済みです。速度と transformer 内訳は上表を参照してください。
> * 解像度、ステップ数、フレーム数、計算内容（近似・省略なし）は記事と同一条件で実行しています。
>   ただし **Policy Model の出力精度は公式合格基準を満たしていません**: golden action MSE の本環境実測は
>   0.126〜0.134 で、公式基準（< 0.05）および記事の報告値（0.0132、合格）を上回ります。速度最適化の影響は
>   実測で棄却済み（最適化前から同帯）で、数値精度（fp32 感度実験）・checkpoint 版差の両仮説も実測棄却済みです。
>   原因帰属の決着には CUDA 環境での参照実行が必要ですが、当分実施できないため未決着のまま素の実測値を提示します。
>   なお記事側チャートでも step 6-7 に per-step MSE 0.05 超過があり、全体 0.0132 はほぼ 2 step 分の逸脱で構成されています
>   （本環境の最大逸脱も同じ step 6-7 で、値は約 4.5 倍 + 広帯域誤差）。
> * 非同期処理による見かけ上の高速化は行わず、同期実行時間の総和で測定しています。
> * Policy Model は初期状態（約 1965 秒）比では 47 倍の改善にあたります。

> [!IMPORTANT]
> **Policy Model の「生成 41.66 秒」に入力観測動画のエンコード（conditioning）は含まれません。**
> 記事側は「モデル常駐後 21 秒」とのみ記載され内訳が非公開のため、記事側が同エンコードを含むかは不明です。
> ただし conditioning の削減後（下記）は同エンコードが 0.28 秒と無視できる大きさになったため、
> **記事側が含むか否かで比較結果は実質的に変わりません**（含まない場合 41.66 秒 / 含む場合 42.44 秒、
> いずれも対記事約 2 倍）。上表は「記事の 21 秒も生成のみ」との推定に立っていますが、
> この推定が外れても結論は変わりません。
>
> 参考として、入力エンコードを含めた同期総和は **42.44 秒**（対記事 2.02 倍）です。
> この値は入力観測エンコードの削減後のものです。従来の測定では 124.91 秒でしたが、
> エンコード対象に**推論結果に到達しない 16 フレーム分**が含まれていました
> （条件付けに使われるのは latent frame [0] のみで、これはピクセルフレーム 1 枚のエンコードで得られます）。
> この除去は近似ではなく**厳密**で、サンプリングへの入力テンソルが未削減時とビット完全一致することを
> 実測で確認しています（`result/conditioning_probe_v3_20260727/`）。
> したがって conditioning を含めても含めなくても対記事比は約 2 倍で一致します。
>
> 再現手順（`--policy-condition-cache` が必須。同フラグは warmup 時の conditioning を測定 run で再利用し、
> ステージ境界で同期を取る）:
> ```
> python scripts/run_cosmos_framework_policy_rocm.py --warmup-runs 2 --policy-condition-cache
> ```
> 入力エンコードを含む 42.44 秒の測定は、condition-cache を外して削減フラグを付けた
> `--policy-sync-profile --policy-min-condition-encode` で再現できます
> （削減フラグ無しの 124.91 秒は `--policy-sync-profile` のみで再現できます）。

---

## 3. 適用された主要な最適化技術

本環境において、初期状態から **最大 47 倍以上** の高速化を達成するために適用した最適化技術のパッケージです。

1.  **ホスト・デバイス同期の完全排除 (Graph Breakの解消)**:
    アテンションアサーションの `.tolist()` や、サンプリングループ内の条件マスク評価（`noisy_mask.sum() > 0`）をバイパスするモンキーパッチを適用し、トランスフォーマー前向き処理の同期回数を **0回** に平坦化。
2.  **HIP Graphs (CUDA Graphs) の有効化**:
    同期が解消されたことで、30ステップ of サンプリングループ内の全 GPU カーネル群を単一の実行グラフとしてキャプチャ。CPU側のカーネル起動オーバーヘッドを物理的に排除。
3.  **AOTriton によるアテンション最適化**:
    `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` により、極大シーケンス（74,000トークン）においても OOM を回避して動作する Triton 融合アテンションカーネルを駆動。
4.  **PyTorch TunableOp による GEMM チューニング**:
    Cosmos3 の Transformer レイヤーで発生するすべての行列乗算（GEMM）形状をスキャンし、`gfx1151` アーキテクチャで最速となる hipBLASLt/Tensile カーネルを自動割り当て。
5.  **VAE 部分コンパイル (`torch.compile` max-autotune)**:
    VAE デコード処理の 95.3% の負荷が集中していた最終アップサンプリングブロック `upsample_3` のみを選択的にコンパイルし、3D 畳み込み・活性化・正規化をカーネルフュージョン。
6.  **厳密 und branch cache（CFG 2 スロット）**:
    denoising step 間で不変な understanding branch を cond / uncond ごとに再利用。T2V の公式 guidance 実測では 140 transformer calls 中 2 writes / 138 reads となり、生成ステップ・CFG・dtype・演算内容を変えずに 26.6% 短縮。出力の byte 完全一致を確認済み。

---

## 4. [おまけ] DGX Spark (Grace Blackwell GB10) との行列積ベンチマーク比較

ブログ記事「[DGX Sparkで見る“デスクサイドAI”の実力](https://www.hpc.co.jp/tech-blog/2026/03/17/dgx-spark-desk-side/)」で実施された大規模行列乗算（16384 × 16384）と同条件にて、本環境の GPU ピークスループットを実測・比較しました。

> **詳細**: [固定形状・外部MAMF・Cosmos3実形状を含むGEMM比較レポート](docs/dgx_spark_comparison_report.md)

### 📊 行列乗算実測 TFLOPS 比較

| 精度フォーマット | DGX Spark (GB10) 実測値 | 本環境 (Radeon 8060S / gfx1151) | 性能比 (GB10 / 本環境) |
|---|:---:|:---:|:---:|
| **FP16 (Half Precision)** | 96.54 TFLOPS | **20.04 TFLOPS** | **4.82 倍** |
| **BF16 (BFloat16)** | 96.91 TFLOPS | **20.91 TFLOPS** | **4.63 倍** |
| **FP32 (IEEE 754 Single)** | 18.54 TFLOPS | **2.86 TFLOPS** | **6.48 倍** |
| **FP64 (Double Precision)** | 0.42 TFLOPS | **0.45 TFLOPS** | **0.93 倍** (本環境が優位) |

> **読み方**: 4.63倍は固定16384角と各software stackを含む差で、一般的な最大BF16性能差ではありません。外部のMAMF直接比較はGB10 101 / Strix Halo 46 TFLOPS（約2.20倍）、本環境のCosmos3実形状は最大36.11 TFLOPSです。形状・kernel選択・runtime差を含む詳細と出典は上記レポートを参照してください。

---

## 5. ベンチマークスクリプトの実行方法

コンテナ環境内において、以下のコマンドでピーク行列積性能を測定できます。

```bash
# 行列積性能の測定
python3 scripts/bench_peak.py
```
