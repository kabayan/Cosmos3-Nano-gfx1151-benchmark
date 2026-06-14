# Cosmos3-Nano ROCm v1.1 高速化 実装・テスト計画

作成日: 2026-06-02

## 位置づけ

ここまでの試行を `v1.0` として固定する。

`v1.0` は、ROCm/gfx1151 環境で Cosmos3-Nano の以下 4 モードを通したベースライン。

| Mode | v1.0 status | v1.0 time | Article time | Ratio |
| --- | --- | ---: | ---: | ---: |
| Text-to-image | success | 974.611 sec | 22 sec | 44.3x slower |
| Text-to-video | success | 483.187 sec | 22 sec | 22.0x slower |
| Image-to-video | success | 166.890 sec | 17 sec | 9.8x slower |
| Policy Model | success | about 1965 sec | 21 sec | 93.6x slower |

`v1.1` の目的は、画質や入力条件を変えず、ROCm 側の実行経路を改善して速度を上げること。

## 参照した一次情報

- ROCm PyTorch compatibility: ROCm 上の PyTorch には `torch.compile` などの core 機能が含まれる。  
  https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html
- PyTorch SDPA: `scaled_dot_product_attention` は fused kernel を自動選択し、`sdpa_kernel()` で backend を制御できる。GQA は experimental で `enable_gqa=True` が必要。  
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html
- AOTriton: PyTorch は ROCm SDPA kernel 経由で AOTriton を利用する。PyTorch/AOTriton の対応関係がある。  
  https://github.com/ROCm/aotriton
- PyTorch TunableOp: ROCm では GEMM を rocBLAS / hipBLASLt から自動選択できる。  
  https://docs.pytorch.org/docs/2.12/cuda.tunable.html  
  https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html
- torch.compile: TorchInductor は AMD GPU でも Triton を使う backend として説明されている。  
  https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler.html
- ROCm PyTorch install/troubleshooting: gfx target が合わない場合は `rocminfo` と `llvm-readobj` で確認し、必要なら `PYTORCH_ROCM_ARCH` を指定して build する。  
  https://rocmdocs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
- rocprof: HIP application の kernel/API trace と counter を取得できる。  
  https://rocmdocs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html

## v1.0 で見えたボトルネック

### Diffusers 経路

T2I/T2V は diffusion step だけでなく、VAE decode / postprocess / export が支配的になっている。

| Case | Total | Observed denoise | Dominant issue |
| --- | ---: | ---: | --- |
| T2I 960x960 35 steps | 974.611 sec | about 201 sec | VAE decode / postprocess が大きい |
| T2V 448x256 24f 35 steps | 483.187 sec | about 52 sec | VAE decode / postprocess / export が大きい |
| I2V 448x256 24f 35 steps | 166.890 sec | about 161 sec | denoise 支配 |

T2I/T2V では `pipe()` 全体の時間だけでは不十分。`denoise`, `VAE decode`, `CPU postprocess`, `export_to_video` を分けて計測する必要がある。

### Policy Model

Policy は `cosmos-framework` 経路で成功したが、公式最適経路ではない。

- `vllm/vllm-omni-rocm:v0.20.0` には `vllm_omni` と `/v1/videos` route 実装はある。
- ただし Cosmos3 固有 action path は確認できなかったため、v1.0 は framework 経路で実行した。
- framework の native attention backend は FlashAttention / NATTEN 前提。
- ROCm/gfx1151 では未導入だったため、検証用 PyTorch SDPA fallback を差し込んだ。
- sampler 30 steps は約 80 sec だったが、前後処理込みで約 1965 sec。

Policy の最大課題は、検証用 fallback から production quality の attention backend へ移すこと。

## 高速化候補

### P0: 計測を分解する

目的:

- `pipe()` 一括時間から脱却し、denoise / decode / export のどこに効いたかを確認する。
- 速度改善の判断を総時間だけにしない。

実装:

- 既存 benchmark script に stage timer を追加する。
- `torch.profiler` または `rocprof` の短縮 run を追加する。
- 出力 summary に `load_seconds`, `pipe_seconds`, `save_seconds`, `total_seconds`, `gpu_memory` を保存する。

判定:

- smoke 条件で全 case が成功する。
- summary JSON に stage 別時間が出る。

### P1: AOTriton / SDPA 経路を検証する

目的:

- local log で出ていた `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を検証し、SDPA/VAE decode に効くか確認する。

実装:

- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` の variant を追加。
- PyTorch SDPA backend が fused kernel を選べているか、短い attention probe と profiler で確認する。
- Policy fallback では `sdpa_kernel()` を使い、math / flash / mem_efficient の可否をログ化する。

期待:

- I2V/Policy sampler の改善余地がある。
- T2I/T2V の VAE decode 側には効かない可能性がある。

リスク:

- gfx1151 では experimental kernel が使えない、または不安定な可能性がある。

### P1: TunableOp を導入する

目的:

- transformer / VAE の linear/GEMM に対して、rocBLAS / hipBLASLt の最適 kernel を選ばせる。

実装:

- まず `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` で untuned GEMM を収集する。
- 次に offline tuning で `tunableop_results.csv` を作る。
- 最後に `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=0` で tuned table を使った本測定を行う。

期待:

- denoise が支配的な I2V で効果が見えやすい。
- Policy sampler にも効果が出る可能性がある。

リスク:

- 初回 tuning は重い。
- PyTorch / ROCm / hipBLASLt version が変わると tuning table は再作成が必要。

### P1: Policy attention fallback を改善する

目的:

- v1.0 の検証用 SDPA fallback を、正確性と速度を両立する実装に近づける。

実装:

- `scripts/run_cosmos_framework_policy_rocm.py` の fallback に以下を追加する。
  - backend 選択ログ
  - causal mask の正当性テスト
  - GQA head repeat の明示 fallback
  - chunk loop の GPU 同期削減
- 可能なら NATTEN / FlashAttention ROCm build を検証する。
- `cosmos-framework` 側の `arch_tag=0` 原因を調べる。

期待:

- Policy の sampler 80 sec はさらに短縮できる余地がある。
- ただし v1.0 では前後処理が支配的なので、総時間改善には decode/保存側も必要。

### P2: torch.compile を限定適用する

目的:

- 静的 shape の transformer blocks / VAE decode に限定して TorchInductor を試す。

実装:

- まず smoke 条件のみ。
- `torch.compile(..., mode="reduce-overhead")` を transformer / VAE に個別適用できるか確認する。
- framework Policy は既存 CLI の `--use-torch-compile` を smoke 条件で戻す。

期待:

- steady-state では改善する可能性がある。

リスク:

- 初回 compile cost が重く、1 回だけの生成では遅くなる可能性がある。
- Python 3.12 + ROCm 7.2 + gfx1151 で graph break / unsupported kernel が出る可能性がある。

### P2: postprocess / export の最適化

目的:

- T2I/T2V の総時間を支配している decode / postprocess / export を削る。

実装候補:

- 画像/動画保存時間を `pipe()` から分離して計測する。
- `export_to_video` と `imageio` の encode 時間を個別計測する。
- intermediate を PNG/JPEG/MP4 で比較し、保存形式に由来する遅延を分離する。
- Diffusers pipeline 内部で VAE decode を chunk / tile / dtype 変更できるか確認する。
- `torch.inference_mode()` の外側明示、CPU tensor conversion の箇所を確認する。

期待:

- T2I/T2V はここが最大改善候補。

### P3: gfx1151 向け PyTorch / ROCm stack の見直し

目的:

- `arch_tag=0` や attention backend 未対応を解消する。

実装候補:

- `rocminfo` で gfx target を確認する。
- `llvm-readobj --offloading` で libtorch HIP code object に gfx1151 が含まれるか確認する。
- 必要なら `PYTORCH_ROCM_ARCH=gfx1151` で PyTorch source build または nightly/ROCm image を比較する。

期待:

- native attention / AOTriton / TorchInductor の挙動改善。

リスク:

- build cost が大きい。
- official support 状態によっては成功しない。

## 追加した実装

速度比較 matrix の dry-run/実行ドライバを追加した。

```text
scripts/run_rocm_speed_matrix.py
scripts/validate_rocm_optimization_primitives.py
```

例:

```bash
python3 scripts/run_rocm_speed_matrix.py --variant v1_0 --case t2i_smoke
python3 scripts/run_rocm_speed_matrix.py --variant aotriton --case tech_validate --execute
python3 scripts/run_rocm_speed_matrix.py --variant aotriton --case t2v_i2v_smoke --execute
python3 scripts/run_rocm_speed_matrix.py --variant tunable_collect --case t2i_smoke --execute
```

対応 variant:

| Variant | Environment |
| --- | --- |
| `v1_0` | v1.0 baseline |
| `aotriton` | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
| `tunable_collect` | TunableOp untuned GEMM collection |
| `tunable_online` | TunableOp online tuning |
| `allocator` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `aotriton_tunable` | AOTriton + TunableOp |

対応 case:

| Case | Purpose |
| --- | --- |
| `tech_validate` | モデル load なしで ROCm 最適化 primitive を検証 |
| `t2i_smoke` | 480x480, 8 steps |
| `t2i_article` | 960x960, 35 steps |
| `t2v_i2v_smoke` | 448x256, 8 frames, 8 steps |
| `t2v_i2v_stage_smoke` | `--stage-profile` 付き T2V/I2V stage 分解 |
| `t2v_i2v_article` | 448x256, 24 frames, 35 steps |
| `policy_article` | action_policy_robot.json |

## テスト計画

### Stage 0: 環境確認

目的:

- GPU / PyTorch / ROCm / SDPA backend の状態を固定する。

テスト:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())
PY
rocminfo | grep gfx
```

合格条件:

- `torch.cuda.is_available()` が true。
- device が v1.0 と同じ。
- gfx target が記録される。

### Stage 1: 技術検証 matrix

目的:

- full benchmark 前に、最適化 primitive がこの ROCm/gfx1151 環境で使えるかを確認する。
- モデル load / 画像生成を行わず、短時間で variant の不成立を除外する。

検証内容:

- PyTorch / HIP / GPU / env flags の記録。
- SDPA default backend の実行。
- SDPA backend 強制 `FLASH_ATTENTION` / `EFFICIENT_ATTENTION` / `MATH` の可否確認。
- GQA SDPA の shape / finite 確認。
- TunableOp API import と `F.linear` probe。
- Policy fallback の varlen + GQA smoke tensor 確認。
- 任意で `torch.compile` 最小 model の初回/2回目実行確認。

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant tunable_collect \
  --variant allocator \
  --case tech_validate \
  --execute
```

直接実行する場合:

```bash
python3 scripts/validate_rocm_optimization_primitives.py \
  --out result/rocm_speed_matrix/local/tech_validate.json
```

合格条件:

- `tech_validate.json` が生成される。
- `cuda_available`, `sdpa_default`, `sdpa_gqa`, `tunable_import`, `tunable_linear_probe`, `policy_fallback_varlen_gqa` が success。
- fused SDPA backend は失敗しても blocker ではない。ただし失敗理由を記録し、AOTriton 採否判断に使う。
- `torch_compile_probe` は任意。失敗しても full benchmark の blocker にはしない。

gate:

- `tech_validate` が失敗した variant は smoke / article benchmark に進めない。
- `tunable_import` が失敗した場合、その variant の TunableOp 系 benchmark は中止する。
- `sdpa_gqa` が失敗した場合、Policy fallback 系 benchmark は中止する。

### Stage 2: smoke matrix

目的:

- 技術検証を通過した variant だけ、短い生成で実際の pipeline が落ちないか確認する。

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant tunable_collect \
  --case t2i_smoke \
  --case t2v_i2v_smoke \
  --execute
```

合格条件:

- 全 case が成功。
- 出力 JSON / media が生成される。
- v1.0 smoke と比較して、明確な品質破綻がない。

### Stage 3: TunableOp table 作成

目的:

- GEMM shape を収集して、offline tuning 可能な状態にする。

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant tunable_collect \
  --case t2i_smoke \
  --case t2v_i2v_smoke \
  --execute
```

次に container 内で:

```python
import torch.cuda.tunable as tunable
tunable.tune_gemm_in_file("/workspace/result/rocm_speed_matrix/tunableop_untuned.csv")
```

合格条件:

- `tunableop_untuned.csv` と `tunableop_results.csv` が生成される。
- tuned table 使用時に出力が成功する。

### Stage 4: article matrix

目的:

- smoke で通った variant だけ記事条件で測る。

実行例:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant aotriton_tunable \
  --case t2i_article \
  --case t2v_i2v_article \
  --execute
```

合格条件:

- T2I/T2V/I2V が成功。
- v1.0 比で 5% 以上の改善なら採用候補。
- 2 回実行してばらつきが 10% 以内なら採用。

### Stage 5: Policy Model

目的:

- Policy の fallback 経路を改善し、成功・速度・action 形状を維持する。

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant aotriton_tunable \
  --case policy_article \
  --execute
```

合格条件:

- `sample_outputs.json` の `status` が `success`。
- action shape が `16 x 10`。
- `vision.mp4` が `640x480`, `17 frames`, `5 fps`。
- v1.0 比で sampler または total が改善。

追加検証:

- golden action と MSE を計算する。
- fallback attention と native backend の出力差分を smoke tensor で比較する。

### Stage 6: 採用判定

採用条件:

- 同一 input / seed / steps / frames。
- 出力 media が生成される。
- 目視で重大な破綻がない。
- v1.0 比で 5% 以上速い。
- 2 回連続で同程度の改善。

不採用条件:

- 生成失敗。
- 出力 shape が変わる。
- action shape が変わる。
- 速度改善が 5% 未満。
- 初回 compile/tuning cost を除いても遅い。

## 優先順位

1. 技術検証 matrix を通し、variant ごとの成立可否を記録する。
2. Stage timer と profiler を追加する。完了: `scripts/benchmark_classmethod_article_t2v_i2v_rocm.py --stage-profile`
3. AOTriton variant を smoke で測る。
4. TunableOp の untuned GEMM を収集する。
5. T2I/T2V の postprocess/export を分解する。
6. Policy fallback の正確性テストを追加する。
7. native attention backend / gfx1151 対応を調査する。
8. torch.compile を限定適用する。

## 期待する v1.1 成果

短期:

- T2I/T2V/I2V のどこが本当に遅いかを stage 別に説明できる。
- full benchmark 前に、AOTriton / TunableOp / SDPA / fallback の成立可否を短時間で判定できる。
- AOTriton と TunableOp の採否を実測で判断できる。
- Policy fallback の正確性と速度リスクを明文化できる。

中期:

- I2V と Policy sampler の改善。
- T2I/T2V の decode/postprocess 改善。
- vLLM-Omni ROCm + Cosmos3 action path の成立可否を再判定。
