# Cosmos3-Nano ROCm 速度向上取り組み分析レポート

本レポートは、AMD GPU（Radeon 8060S / gfx1151）および ROCm 7.2 環境における `nvidia/Cosmos3-Nano` の動作検証および速度向上（最適化）への取り組み計画を分析したものです。

---

## 1. プロジェクトの概要と検証状況
本プロジェクトは、NVIDIA製マルチモーダルモデル **Cosmos3-Nano** を AMD の ROCm 環境で動作させ、実用的な速度まで最適化することを目的としています。
現在までに以下の主要な機能が ROCm 7.2 上で動作検証されています（詳細は [cosmos3-nano-rocm-work-summary-and-setup.md](file:///home/kabayan/workspace/cosmos3/docs/cosmos3-nano-rocm-work-summary-and-setup.md) を参照）。

*   **動作成功項目**: Text-to-Image (T2I), Text-to-Video (T2V), Image-to-Video (I2V), vLLM Reasoner（画像・動画 of テキスト変換）, Policy Model（ビデオ＋アクション出力）

---

## 2. 当初の課題とボトルネック (v1.0)
初期検証（v1.0）時点では、NVIDIA (CUDA) 環境の公式ベンチマーク値と比較して非常に低速でした。
*   **T2I**: 974.6秒 (論文値22秒に対し 44.3倍遅い)
*   **T2V**: 483.2秒 (論文値22秒に対し 22.0倍遅い)
*   **Policy**: 約1965秒 (論文値21秒に対し 93.6倍遅い)

### 主なボトルネック原因:
1.  **Cold Start (初回実行) オーバーヘッド**: 初回実行時に MIOpen/Tensile が最適カーネルの探索 (Find) を行うため、数十秒〜数分の巨大な遅延が発生する。
2.  **非効率な GEMM カーネルの選択**: AMD の `gfx1151` アーキテクチャに対し、PyTorch のデフォルト設定が Cosmos モデルの形状に合わない非効率な Tensile GEMM カーネルを選択していた。
3.  **注意機構 (Attention) バックエンドの欠如**: ROCm 向け NATTEN や FlashAttention が未導入であり、効率的な fused attention が効いていなかった。

---

## 3. 速度向上の取り組みと成果 (v1.1 〜 v2.3)

ボトルネックの特定に基づき、以下の高速化施策が実施・検証されました。

### ① AOTriton の導入 (Attention 最適化)
*   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を適用することで、PyTorch が SDPA (Scaled Dot Product Attention) の fused kernel を自動選択できるようにしました。

### ② PyTorch TunableOp による GEMM の最適化 (T2V / I2V)
*   [cosmos3-nano-rocm-gemm-kernel-selection-deep-dive.md](file:///home/kabayan/workspace/cosmos3/docs/cosmos3-nano-rocm-gemm-kernel-selection-deep-dive.md)
*   **内容**: `PYTORCH_TUNABLEOP_ENABLED=1` を使用し、Cosmos モデルの Transformer が使用する GEMM (行列乗算) 形状に対して最適な Tensile カーネルをオフライン/オンライン探索させました。
*   **成果**: 
    *   T2V において Transformer Forward の処理時間を **41.2秒から 27.0秒へ高速化 (1.53倍)**。
    *   I2V において Transformer Forward の処理時間を **89.0秒から 71.6秒へ高速化 (1.24倍)**。

### ③ Policy Model における Condition Cache の導入
*   [cosmos3-rocm-v2_3-policy-speedup-plan.md](file:///home/kabayan/workspace/cosmos3/docs/cosmos3-rocm-v2_3-policy-speedup-plan.md)
*   **内容**: 入力テキストやパッキングインデックス等の不変なメタデータをキャッシュし、各サンプリングステップでの重複計算を削減。
*   **成果**: Policy Model のトータル実行時間を **303.3秒から 147.8秒へ半減 (約 2 倍高速化)**。

### ④ VAE 3D Conv デコード処理の深掘りと過去の誤解の解消
*   [cosmos3-rocm-v2_3-policy-decoder-block-and-velocity-rocprof-results.md](file:///home/kabayan/workspace/cosmos3/docs/cosmos3-rocm-v2_3-policy-decoder-block-and-velocity-rocprof-results.md)
*   [cosmos3-rocm-v2_3-vae-single-conv-descriptor-probe-improvement.md](file:///home/kabayan/workspace/cosmos3/docs/cosmos3-rocm-v2_3-vae-single-conv-descriptor-probe-improvement.md)
*   **内容**: VAE デコード処理 (97.1秒) のうち、なんと **95.3% (92.5秒)** が最終のアップサンプリングブロック `upsample_3` に集中していることを突き止めました。
*   **誤解の解消**: 過去には `channels_last_3d` に変換することで高速化されると仮定されていましたが、Warm 状態の厳密な測定により、定常状態では baseline (通常チャネル順) の方がむしろ **5倍以上高速** であることが実証されました（従来の channels_last_3d の優位性は Cold-start 時の探索時間を一時的に回避したことによる見かけ上のアーティファクトでした）。

---

## 4. 意思決定と調査結果（2026年6月12日更新）

### ① 等価性（完全一致）の許容度と MIOpen 設定
*   **決定**: 出力の「完全一致 (bit-level / hash-level)」は不要とし、ある程度の出力誤差（アクション値の数%以内の誤差など）があっても生成物のクオリティが損なわれなければ許容する。
*   **影響**: これにより、Cold-start の探索時間を極小化できる `MIOPEN_FIND_MODE=FAST` が本番稼働において非常に有効な選択肢となります。

### ② チューニングの更新サイクル
*   **決定**: **YAGNI（You Aren't Gonna Need It）**の方針を適用。自動更新や複雑なパイプラインは構築せず、当面は手動で作成した静的なチューニングテーブル（`tunableop_results0.csv`）を再利用するシンプルな構成（`aotriton_tuned` モード等）で運用します。

### ③ Cosmos3 本体の更新確認と最新追従
*   NVIDIA は Cosmos に関連するリポジトリを統合し、新設された **`NVIDIA/Cosmos` (https://github.com/NVIDIA/Cosmos)** を中央リポジトリとして移行しました。
*   **対応**: ワークスペース内に最新の統合リポジトリ `NVIDIA/Cosmos` を `third_party/cosmos` として clone し、最新のレシピやドキュメントを参照可能にしました。

### ④ ローカルコミットの実行と最新 upstream へのマージ
1.  **diffusers**:
    *   キャッシュ機構（`_und_branch_cache`）に関する [transformer_cosmos3.py](file:///home/kabayan/workspace/cosmos3/third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py) の変更をコミットしました。
    *   さらに、`huggingface/diffusers.git` (origin/main) の最新コミットを fetch してローカルコミットと正常にマージし、最新の Diffusers に追従させました。
2.  **rocm-libraries**:
    *   MIOpen の Workspace 制限緩和（BF16/FP16）に関する [gemm_common.cpp](file:///home/kabayan/workspace/cosmos3/third_party/rocm-libraries-rocm-7.2.0/projects/miopen/src/solver/conv/gemm_common.cpp) の変更を、新ブランチ `local-opt` 上にコミットして保存しました。
