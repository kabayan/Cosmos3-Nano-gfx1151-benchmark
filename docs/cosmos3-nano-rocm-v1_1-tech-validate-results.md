# Cosmos3-Nano ROCm v1.1 技術検証結果

実施日: 2026-06-02

## 目的

full benchmark 前に、ROCm 高速化 primitive がこの環境で成立するか確認する。

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant tunable_collect \
  --variant allocator \
  --case tech_validate \
  --execute
```

実行対象:

```text
scripts/validate_rocm_optimization_primitives.py
```

## 環境

```text
torch: 2.9.1+rocm7.2.0.git7e1940d4
HIP: 7.2.26015-fc0010cf6a
device: AMD Radeon Graphics
VRAM reported by torch: 120.0 GiB
```

## 結果サマリ

全 variant で `tech_validate.json` は生成され、gate 上の必須項目は成功した。

| Variant | Overall | SDPA default | Flash SDPA forced | Efficient SDPA forced | SDPA GQA | TunableOp import | Policy fallback |
| --- | --- | ---: | --- | --- | ---: | --- | ---: |
| `v1_0` | ok | 0.203339 sec | fail | fail | 0.000276 sec | ok | 0.058702 sec |
| `aotriton` | ok | 0.018353 sec | ok / 0.000305 sec | ok / 0.000145 sec | 0.000353 sec | ok | 0.026424 sec |
| `tunable_collect` | ok | 0.353915 sec | fail | fail | 0.000234 sec | ok | 0.059422 sec |
| `allocator` | ok | 0.202114 sec | fail | fail | 0.000243 sec | ok | 0.058558 sec |

## 個別所見

### AOTriton

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を入れた `aotriton` variant だけ、強制指定した Flash/Efficient SDPA backend が成功した。

v1.0 では以下が失敗した。

```text
sdpa_backend_flash_attention: RuntimeError('No available kernel. Aborting execution.')
sdpa_backend_efficient_attention: RuntimeError('No available kernel. Aborting execution.')
```

AOTriton 有効時:

```text
sdpa_backend_flash_attention: success
sdpa_backend_efficient_attention: success
```

このため、次の smoke benchmark に進める最有力 variant は `aotriton`。

### TunableOp

`torch.cuda.tunable` の import と `F.linear` probe は成功した。

ただし `tunable_collect` は SDPA backend には影響しない。full benchmark 前に、TunableOp は GEMM shape 収集・tuning table 作成の目的で扱う。

### allocator

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` は以下の警告を出した。

```text
expandable_segments not supported on this platform
PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
```

このため、allocator variant は full benchmark 候補から外す。

### Policy fallback

Policy fallback の varlen + GQA smoke は全 variant で成功した。

```text
shape: [1, 96, 32, 64]
```

AOTriton 有効時は probe 時間が短くなった。

```text
v1_0:    0.058702 sec
aotriton: 0.026424 sec
```

ただしこれは小さい synthetic tensor の結果であり、Policy full run の改善を保証するものではない。

## Gate 判定

### smoke benchmark に進める

- `v1_0`
- `aotriton`
- `tunable_collect`

### 条件付きで進める

- `tunable_online`
- `aotriton_tunable`

理由: 今回は `tech_validate` を直接実行していないが、構成要素である AOTriton と TunableOp import は成立している。TunableOp table 作成後に使う。

### full benchmark 候補から外す

- `allocator`

理由: `expandable_segments` がこの platform で未対応。

## 次の実行

推奨 smoke:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant tunable_collect \
  --case t2i_smoke \
  --case t2v_i2v_smoke \
  --execute
```

推奨 TunableOp 収集:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant tunable_collect \
  --case t2i_smoke \
  --case t2v_i2v_smoke \
  --execute
```

## 出力ファイル

```text
result/rocm_speed_matrix/v1_0/tech_validate/tech_validate.json
result/rocm_speed_matrix/aotriton/tech_validate/tech_validate.json
result/rocm_speed_matrix/tunable_collect/tech_validate/tech_validate.json
result/rocm_speed_matrix/allocator/tech_validate/tech_validate.json
```
