# Cosmos3-Nano Policy Model さらなる高速化余地の再調査レポート（2026-07-26）

本ドキュメントは、最終報告（[cosmos3_rocm_policy_optimization_final_report.md](cosmos3_rocm_policy_optimization_final_report.md)、総実行時間 **41.66 秒**、2026-06-14）以降のソフトウェアスタック更新を踏まえ、**生成条件（30 steps / 640x480 / 17f）と同期総和測定を維持したまま**の残余高速化余地を再調査した結果をまとめたものです。

- 調査日: 2026-07-26（最終報告から約 6 週間後）
- 関連判断: DLS-001（調査スコープの採用。`.claude/.dls/active.md`）
- 調査方法: ローカルコードベース確認 + 5 観点の並列 Web 調査（ROCm リリース / AOTriton / PyTorch / vLLM / 帯域削減 attention 手法）

---

## 1. 現状の到達点と既知の限界（前提の再掲）

> [!CAUTION]
> **2026-07-28 訂正**: 下表の「対論文比 1.44 倍（達成済み）」および「論文値（8.0 秒）」は、当時の最終報告が
> 用いていた基準値「論文値 29.00 秒」（サンプリング 21.00 秒 + デコード 8.00 秒）に基づく記載ですが、
> **この基準値には一次出典がありません**。一次情報は参照記事の「モデル常駐後 **21 秒**で出力」（総所要時間、
> 内訳非公開）のみであり、記事 21 秒基準では **1.98 倍で目標 1.5 倍以内は未達**です（DLS-006 で是正済み。
> その後 DLS-016 で合否軸自体を「価格差 2.0 倍以内」に変更し、生成 1.98x は新基準の内側）。
> VAE デコードの「論文値 8.0 秒」も同様に出所不明で、記事側の内訳は非公開です。
> 下表は調査時点のスナップショットとして原文のまま残します。

| ステージ | 実測 | 内訳 |
|---|---:|---|
| 推論総実行時間 | **41.66 秒** | 対論文比 1.44 倍（目標 1.5 倍以内を達成済み） |
| ┗ サンプリング | 33.84 秒 | **1.128 秒/step × 30 steps**。うち約 1.0 秒/step が 74k トークン FlashAttention の KV 再ロードによるメモリ帯域（280 GB/s）飽和と診断済み |
| ┗ VAE デコード | 7.49 秒 | 論文値（8.0 秒）超え達成済み |

[cosmos3_rocm_performance_limit_reassessment.md](cosmos3_rocm_performance_limit_reassessment.md) の結論:
**1.127 秒/step は「当時のスタック（ROCm 7.2 / AOTriton 0.11b）における」メモリ帯域の物理限界**。本調査はこの括弧内の前提が 6 週間で変化したかの検証である。

### モデル形状の確定情報（本調査でローカル確認）

| 項目 | 値 | 出典 |
|---|---|---|
| num_attention_heads | 32 | `~/.cache/huggingface/.../Cosmos3-Nano/transformer/config.json` |
| **head_dim** | **128** | 同上（後述の AOTriton チューニング DB 適用可否を左右） |
| hidden_size | 4096 | 同上 |
| 現行 attention 経路 | `torch.ops.aten._flash_attention_forward`（AOTriton FA varlen） | `scripts/run_cosmos_framework_policy_rocm.py:70` |

---

## 2. 環境更新の進展（2026-06-14 以降）

### 2.1 ROCm: 7.2 → 7.14.0（採番変更、「7.3/7.4」は存在しない）

| 事実 | gfx1151 への影響 |
|---|---|
| ROCm はバージョン体系を変更し、TheRock ベースの **ROCm 7.14.0（2026-07-15 GA）** に移行 | — |
| **gfx1151（Ryzen AI MAX）が preview 表記なしの正式サポート入り**（7.2 では Preview / PyTorch on Linux のみ） | バグ修正の恩恵が届きやすくなる（例: 7.13 系譜で gfx1151 batchnorm inline asm エラー修正） |
| **CK FMHA（Composable Kernel FlashAttention forward）が RDNA3 系で利用可能に**（7.13 preview 系譜） | ⚠️ RDNA3.5（gfx1151）で実際に選択されるかは公式明記なし。**実測検証が必要**。ROCm/flash-attention の対応表も「RDNA 3/4」で 3.5 は非明記 |
| gfx1151 の hipBLASLt / GEMM チューニングは **AMD がまだ shape ログ収集段階**（TheRock #2591、Open） | ライブラリ更新による GEMM 高速化は期待薄。**TunableOp による自前チューニングは引き続き正解** |
| MIOpen は 3.5.1 のまま据え置き。gfx1151 向け Conv3d BF16 改善なし | VAE 側の伸び代はライブラリ更新からは出ない |
| 未解決の既知問題: Strix Halo が compute 負荷時に低クロック張り付き（ROCm/ROCm #5750） | 該当していれば全ステージに影響。**ローカルで clock 挙動の確認価値あり** |

出典: [ROCm release history](https://rocm.docs.amd.com/en/latest/release/versions.html) / [7.14.0 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) / [7.13.0-preview release notes](https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html) / [TheRock #2591](https://github.com/ROCm/TheRock/issues/2591) / [ROCm #5750](https://github.com/ROCm/ROCm/issues/5750)

### 2.2 AOTriton / PyTorch: 0.11b → 0.12b、torch 2.9.1 → 2.13.0

| 事実 | gfx1151 への影響 |
|---|---|
| **AOTriton 0.12b（2026-05-18）で gfx1100/gfx1151 が experimental を卒業**。gfx115x（RDNA3.5）は独立配布 tarball に分割 | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` フラグが不要になる正式経路 |
| **PyTorch 2.13.0（2026-07-08）が AOTriton 0.12b を同梱**し、gfx1151 SDPA が stable 昇格。公式 wheel は `--index-url .../whl/rocm7.2`（現行 ROCm 7.2 と同世代）で、gfx1151 ネイティブカーネル内蔵（PR #167299） | **OS/ROCm を変えずに torch だけ 2.13.0 へ更新して検証可能** |
| torch 2.10〜2.13 の RDNA 関連修正: **wave32 GPU で Inductor の `num_warps` が誤って半減するバグ修正**（2.13）、gfx1150/1151 の hipBLASLt GEMM リスト入り（2.10）、Origami（解析的 GEMM 構成選択）、pointwise ヒューリスティクス改善 | gfx1151 は wave32 なので `num_warps` 修正は compiled カーネル（非 GEMM 部）に直接効く可能性 |
| **AOTriton PR #200（2026-07-13 merge、未リリース）**: gfx1150/1151 の flash tuning DB 追加。大きい seqlen でピーク +97% | ❌ **hdim=64 限定のため Cosmos3-Nano（head_dim=128）には適用されない**。hdim=128 の gfx115x tuning DB は未整備 → attention 単体の劇的改善は次期リリースでも期待できない |
| 0.12b 時点の tuning DB 更新は gfx942/950/1100/1201 のみ（gfx1151 なし）。SplitKV forward 等の長シーケンス構造改善もなし | torch 2.13 更新による attention 高速化は「dispatcher/コンパイラ更新分」にとどまる見込み（小幅） |
| CK FA は gfx1151 非明記、ROCm/flash-attention の Triton 版は aiter 移行 regression 中（gfx1151 で 2.2〜3.7 倍遅、未解決） | **AOTriton (SDPA) が引き続き gfx1151 の本命**という現行構成の妥当性を裏付け |

出典: [aotriton 0.12b](https://github.com/ROCm/aotriton/releases/tag/0.12b) / [aotriton PR #200](https://github.com/ROCm/aotriton/pull/200) / [pytorch v2.13.0](https://github.com/pytorch/pytorch/releases/tag/v2.13.0) / [PyTorch 2.13 blog](https://pytorch.org/blog/pytorch-2-13-release-blog/) / [flash-attention #2392](https://github.com/Dao-AILab/flash-attention/issues/2392)

---

## 3. vLLM 統合仮説（v2.5 提案書の「本命」）の再評価 → 棄却

[cosmos3_rocm_policy_optimization_proposals.md](cosmos3_rocm_policy_optimization_proposals.md) は「vllm-cosmos3 統合による PagedAttention 移植」を本命としていたが、本調査により**構造的に無効**と判定した。

1. **ローカル事実**: `temp_src/packages/vllm-cosmos3` は NVIDIA 公式の **Reasoner（VLM テキスト生成）用プラグイン**（`Cosmos3ReasonerForConditionalGeneration` を登録）であり、Policy Model の DiT サンプリングに使う実装ではない。
2. **構造的理由**: PagedAttention の利益源は (a) 多数同時リクエストの KV キャッシュ断片化解消、(b) autoregressive decode の paged KV 読み出し、(c) prefix キャッシュ共有。本ワークロードは「バッチ 1・固定長 74k・毎ステップ全再計算・decode ゼロ」の 100% prefill 相当であり、**いずれの利益源も存在しない**。
3. **最も強い反証**: vLLM 公式の DiT 対応（[vllm-omni](https://github.com/vllm-project/vllm-omni)、2026-07）は **PagedAttention を使っておらず**、Diffusers 比 1.26x の源泉は演算子融合と flash-attention バックエンド再利用（[vLLM-Omni 論文](https://arxiv.org/html/2602.02204v1)）。その flash-attention に相当するものは本環境では AOTriton FA として適用済み。
4. **RDNA 事情**: vLLM 自体は gfx1151 公式サポート入り（ROCm 7.0.2+）だが、RDNA で使える attention は Triton フォールバック系のみで、AITER の看板性能（2.7〜4.4x）は CDNA の ASM/CK カーネル由来。gfx1151 は AITER で Experimental 扱い。

**結論**: vLLM 統合は Policy Model サンプリング高速化の手段として**棄却**。将来 vllm-omni の演算子融合が RDNA で実証されたときのみ再評価する。

出典: [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) / [ROCm attention backend blog](https://vllm.ai/blog/2026-02-27-rocm-attention-backend) / [AITER](https://github.com/ROCm/aiter) / [vllm-omni](https://github.com/vllm-project/vllm-omni)

---

## 4. 帯域削減 attention / ステップ間キャッシュ手法の gfx1151 実用性

診断済みボトルネック（attention の KV 再ロードによる帯域飽和）を**転送量の削減**で攻める選択肢の評価。

### 4.1 ハードウェア前提

- **gfx1151（RDNA3.5）に FP8 行列ハードウェアは存在しない**（FP8 WMMA は RDNA4 = gfx12 で新設）。FP8 エミュレーションは INT8 比 3.7 倍遅の実測あり → **FP8 attention 系（SageAttention2 等）は不可。8bit 化するなら INT8 一択**
- INT8 WMMA は RDNA3.5 にハードウェアあり

### 4.2 手法別評価

| 手法 | gfx1151 実用性 | 期待効果 | 品質影響 / 制約整合性 |
|---|---|---|---|
| **TeaCache**（ステップ間の特徴冗長性でブロック計算を再利用） | ◎ 純 PyTorch・カーネル非依存。**TeaCache4Cosmos が公式サポート**（Cosmos 系 DiT 向け設定が存在） | サンプリング **1.4〜1.6 倍**（保守的閾値）〜2 倍。総実行時間 41.66s → **約 29〜32 秒** 相当 | 出力が微小に変化する。「数%誤差でもクオリティ非劣化なら許容」の既存決定（2026-06-12 等価性判断）の範囲内かは**要検証**。また「30 steps という条件は維持されるが計算内容は変わる」ため、対論文同一条件比較の主張には注記が必要（§6 参照） |
| **SageAttention v1（INT8 QK、Triton 版）** | ○ RDNA3（RX7900XTX + ROCm7）で動作実績あり（Flux 16% 改善報告）。**gfx1151 での動作報告はなし、要移植検証** | K の INT8 化で attention の KV 転送 25〜30% 減。attention 部 1.2〜1.5 倍 → ステップ 1.128s → 0.9〜1.0s 程度の試算 | end-to-end メトリクス劣化なしと報告（ICLR2025、動画 DiT 含む）。TeaCache と独立に効くため併用可 |
| **FlexAttention + BlockMask 自作 block-sparse** | △ FlexAttention 自体は ROCm 公式サポート。sparse 設計は自作 | sparsity に比例して KV ロード削減 | 開発コスト大。SpargeAttn / STA（CUDA/H100 専用で ROCm 不可）の代替として唯一の ROCm 現実路線 |
| SpargeAttn / STA（sparse attention） | ✗ CUDA≥12 / H100 ThunderKittens 専用 | — | — |
| FP8 attention（SageAttention2 等） | ✗ FP8 ハードなし、エミュは逆効果 | — | — |

出典: [TeaCache](https://github.com/ali-vilab/TeaCache) / [SageAttention](https://github.com/thu-ml/sageattention) / [RDNA3 での SageAttention 動作報告](https://github.com/guinmoon/rocm7-triton-flash-attention-sage-attention-bnb) / [gfx1151 FP8/INT8 実測](https://github.com/lhl/fsr4-rdna3-optimization) / [RDNA4 WMMA](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) / [SpargeAttn](https://github.com/thu-ml/SpargeAttn) / [FlexAttention ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html)

---

## 5. 推奨アクション（優先順位付き）

コスト（検証工数・リスク）対効果で並べる。①②は独立に実行可能、③④⑤は変化があった場合のみ深掘り。

| # | アクション | 期待効果 | コスト / リスク | 制約整合性 |
|---|---|---|---|---|
| ① | **torch 2.13.0（公式 rocm7.2 wheel）への更新検証**: AOTriton 0.12b + wave32 `num_warps` 修正 + hipBLASLt GEMM リストの効果を同一ベンチで実測。TunableOp テーブルは再チューニング | 小〜中（数%。hdim=128 の tuning DB 未整備のため attention の劇的改善はなし。Inductor 側の回復分が主） | 低〜中（wheel 差し替えのみ。ただし torch.compile / HIP Graphs / モンキーパッチの再検証必須） | ✅ 完全整合（計算内容不変） |
| ② | **低クロック張り付き問題（ROCm #5750）の該当確認**: サンプリング中の GPU クロックをモニタし、張り付きがあれば電源設定 / カーネルパラメータで対処 | 該当していれば大、していなければゼロ | 極小（観測のみ） | ✅ 完全整合 |
| ③ | **TeaCache4Cosmos の品質検証付き導入**: 保守的閾値から開始し、生成物のクオリティ比較（既存の等価性判断①の基準）を通す | **大（サンプリング 1.4〜1.6 倍 → 総実行時間 29〜32 秒圏）** | 中（導入は容易、品質検証に工数。同一条件比較の主張に注記が必要 → §6） | ⚠️ 要判断 |
| ④ | **ROCm 7.14 環境での CK FMHA 選択可否の実測**: gfx1151 で CK FMHA forward が動く/速いかをプローブ（別コンテナで安全に検証可能） | 不明（動けば attention 経路の代替候補） | 中（環境構築。7.14 は TheRock 移行直後で安定性リスク） | ✅ 完全整合 |
| ⑤ | **SageAttention v1 Triton（INT8 QK）の gfx1151 移植検証** | 中（ステップ 1.128s → 0.9〜1.0s 試算） | 高（gfx1151 未検証、Triton カーネル調整の可能性、品質検証も必要） | ⚠️ 要判断（量子化誤差、等価性判断①の範囲内か） |

> **2026-07-28 検証追記（DLS-022）**: ①は隔離 image で実測し、不採用とした。torch 2.13.0 + AOTriton 0.12.0 は同一 seed の T2V 出力が現行 2.9.1 と一致せず、decoded 21 frame 全てが相違した（PSNR 28.41 dB / SSIM 0.9488）。本プロジェクトの精度不変条件は diffusers 経路の hash 一致なので、速度調律前に不合格が確定した。旧 TunableOp 表も rocBLAS validator 不一致で流用不可だったため、現行 2.9.1 stack を維持する。

**やらないこと（棄却）**:
- vLLM 統合（§3 の構造的理由により棄却）
- FP8 attention（ハードウェア不在）
- SpargeAttn / STA の直接利用（CUDA 専用）
- MIOpen / hipBLASLt の更新待ちによる VAE・GEMM 改善（gfx1151 チューニングは AMD 側が着手前）

---

## 6. 「同一条件」制約と TeaCache 系手法の整合性について

本プロジェクトの価値は「生成クオリティに関わる条件を一切変更しない対論文比較」にある。TeaCache / INT8 attention は **steps・解像度・フレーム数を変えない**が、計算内容を近似するため出力が微小に変化する。

- 2026-06-12 の意思決定①（[cosmos3_rocm_optimization_analysis.md §4](cosmos3_rocm_optimization_analysis.md)）は「bit 一致は不要、クオリティ非劣化なら数%誤差を許容」を既に採用している
- ただし対外的な比較主張（README / 記事）では「近似キャッシュ併用」の注記を付けるのが誠実

> [!IMPORTANT]
> **確定判断（2026-07-26、DLS-003）**: 計算省略系（TeaCache / INT8 attention 等）は
> 「元記事と同じ計算内容での比較が必須」という理由により**本線から除外**が確定した。
> 等価性判断①は「同じ計算の数値誤差」の許容であり、計算内容の省略はその範囲外。
> TeaCache は速度成果としてではなく、**出力が非適用時とどの程度異なるかの品質差
> 定量評価のみ**を別トラックで実施する（§5 の ③⑤ は本線候補から品質評価タスクへ変更）。

---

## 7. 結論

1. **「1.127 秒/step = 物理限界」の結論は、計算を省かない限り 2026-07 時点でも実質有効**。6 週間のスタック進展（ROCm 7.14 / AOTriton 0.12b / torch 2.13）に、hdim=128 の 74k トークン attention を直接高速化する材料は入っていない（gfx1151 の tuning DB は hdim=64 のみ、CK FMHA は RDNA3.5 非明記）。
2. **torch 2.13.0 更新は DLS-022 の出力非等価により不採用**。クロック張り付きは本環境で非該当を確認済みで、stable stack 更新から確実に取れる小さな伸び代は現時点で残っていない。
3. **大きな伸び代（41.66 秒 → 30 秒圏）は「計算の省略」からのみ得られる**。第一候補は Cosmos 公式サポートのある TeaCache で、品質検証と「同一条件」主張の整理が導入条件。
4. v2.5 提案書の「vLLM 統合が本命」仮説は本調査で**棄却**（PagedAttention の利益源が本ワークロードに存在しない）。

---

## 8. 調査ソース一覧

- ROCm: [release versions](https://rocm.docs.amd.com/en/latest/release/versions.html), [7.14.0 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html), [7.13.0-preview notes](https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html), [TheRock #2591](https://github.com/ROCm/TheRock/issues/2591), [ROCm #5750](https://github.com/ROCm/ROCm/issues/5750), [ROCm #5643](https://github.com/ROCm/ROCm/issues/5643), [MIOpen CHANGELOG](https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md)
- AOTriton: [releases](https://github.com/ROCm/aotriton/releases), [0.12b](https://github.com/ROCm/aotriton/releases/tag/0.12b), [0.13b](https://github.com/ROCm/aotriton/releases/tag/0.13b), [PR #200](https://github.com/ROCm/aotriton/pull/200), [ROCm #6034](https://github.com/ROCm/ROCm/issues/6034), [ROCm #5404](https://github.com/ROCm/ROCm/issues/5404)
- PyTorch: [v2.13.0 release](https://github.com/pytorch/pytorch/releases/tag/v2.13.0), [2.13 blog](https://pytorch.org/blog/pytorch-2-13-release-blog/), [2.12 blog](https://pytorch.org/blog/pytorch-2-12-release-blog/), [PR #167299](https://github.com/pytorch/pytorch/pull/167299), [Phoronix 2.10](https://www.phoronix.com/news/PyTorch-2.10-Released), [Radeon PyTorch install](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- vLLM / AITER: [GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html), [releases](https://github.com/vllm-project/vllm/releases), [ROCm attention backend blog](https://vllm.ai/blog/2026-02-27-rocm-attention-backend), [Triton backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive), [AITER](https://github.com/ROCm/aiter), [vllm-omni](https://github.com/vllm-project/vllm-omni), [vLLM-Omni 論文](https://arxiv.org/html/2602.02204v1), [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes), [llm-tracker Strix Halo](https://llm-tracker.info/_TOORG/Strix-Halo)
- 帯域削減手法: [TeaCache](https://github.com/ali-vilab/TeaCache) ([論文](https://arxiv.org/abs/2411.19108)), [SageAttention](https://github.com/thu-ml/sageattention) ([v1 論文](https://arxiv.org/abs/2410.02367), [v2 論文](https://arxiv.org/abs/2411.10958)), [SpargeAttn](https://github.com/thu-ml/SpargeAttn), [STA / FastVideo](https://github.com/hao-ai-lab/FastVideo/blob/main/csrc/sliding_tile_attention/README.md), [PAB / VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [ToCa](https://arxiv.org/abs/2410.05317), [FORA](https://arxiv.org/abs/2407.01425), [RDNA4 WMMA (GPUOpen)](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/), [Chips and Cheese RDNA4 LLVM](https://chipsandcheese.com/p/examining-amds-rdna-4-changes-in-llvm), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [flash-attention #2392](https://github.com/Dao-AILab/flash-attention/issues/2392), [ROCm/flash-attention](https://github.com/ROCm/flash-attention)
- ローカル確認: `scripts/run_cosmos_framework_policy_rocm.py`, `temp_src/packages/vllm-cosmos3/`, `~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/.../transformer/config.json`, `cosmos3_rocm_pip_freeze.txt`
