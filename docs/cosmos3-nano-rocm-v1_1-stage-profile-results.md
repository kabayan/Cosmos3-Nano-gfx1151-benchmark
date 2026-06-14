# Cosmos3-Nano ROCm v1.1 stage profile 結果

実施日: 2026-06-02

## 目的

T2V smoke benchmark が AOTriton 有効時でも総時間改善しなかったため、`pipe()` 内の処理を分解し、支配的な stage を確認する。

## 追加実装

`scripts/benchmark_classmethod_article_t2v_i2v_rocm.py` に `--stage-profile` を追加した。

計測対象:

```text
transformer_forward
vae_decode
video_postprocess
export_to_video
```

`run_rocm_speed_matrix.py` には診断用 case を追加した。

```text
t2v_i2v_stage_smoke
```

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --case t2v_i2v_stage_smoke \
  --execute
```

## 条件

```text
height: 256
width: 448
frames: 8
fps: 8
steps: 8
guidance: 1.0
```

## 結果

### T2V

```text
total pipe seconds: 433.401
export_to_video: 0.246
```

stage profile:

| Stage | Calls | Seconds |
| --- | ---: | ---: |
| transformer_forward | 8 | 7.570 |
| vae_decode | 1 | 368.638 |
| video_postprocess | 1 | 0.026 |
| unattributed_pipe | - | 57.167 |

内訳:

```text
vae_decode: 85.0% of pipe time
transformer_forward: 1.7% of pipe time
unattributed_pipe: 13.2% of pipe time
```

### I2V

```text
total pipe seconds: 31.867
export_to_video: 0.072
```

stage profile:

| Stage | Calls | Seconds |
| --- | ---: | ---: |
| transformer_forward | 8 | 30.394 |
| vae_decode | 1 | 1.012 |
| video_postprocess | 1 | 0.006 |
| unattributed_pipe | - | 0.455 |

内訳:

```text
transformer_forward: 95.4% of pipe time
vae_decode: 3.2% of pipe time
unattributed_pipe: 1.4% of pipe time
```

## 結論

T2V と I2V は同じ 448x256 / 8 frames / 8 steps でも、支配 stage が大きく異なる。

- T2V: VAE decode 支配。
- I2V: transformer denoise 支配。

## AOTriton 追加検証

追加で同じ条件を `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 付きで実行した。

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case t2v_i2v_stage_smoke \
  --execute
```

### v1.0 vs AOTriton

| Case | Variant | Total pipe | Transformer | VAE decode | Postprocess | Unattributed | Export |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | `v1_0` | 433.401 | 7.570 | 368.638 | 0.026 | 57.167 | 0.246 |
| T2V | `aotriton` | 433.241 | 7.193 | 368.400 | 0.009 | 57.639 | 0.107 |
| I2V | `v1_0` | 31.867 | 30.394 | 1.012 | 0.006 | 0.455 | 0.072 |
| I2V | `aotriton` | 18.677 | 17.338 | 1.012 | 0.005 | 0.322 | 0.075 |

差分:

```text
T2V total:       433.401 -> 433.241 sec  (-0.04%)
T2V vae_decode: 368.638 -> 368.400 sec  (-0.06%)
I2V total:        31.867 ->  18.677 sec (-41.4%)
I2V transformer:  30.394 ->  17.338 sec (-43.3%)
```

### AOTriton 追加検証の結論

- T2V の VAE decode には AOTriton はほぼ効いていない。
- I2V の transformer には AOTriton が明確に効いている。
- T2V を速くするには VAE decode 単体の追加調査が必要。

したがって、次の最適化方針は分ける。

### T2V

T2V は AOTriton / SDPA よりも VAE decode 最適化を優先する。

候補:

- VAE decode の chunk / tile 設定確認。
- VAE decode dtype / attention backend の確認。
- VAE decode の profiler trace を取る。

### I2V

I2V は transformer が支配的なので、AOTriton の article benchmark へ進める価値がある。

候補:

- `v1_0` vs `aotriton` の I2V article benchmark。
- transformer forward の fused SDPA backend 有効化確認。

## 出力ファイル

```text
result/rocm_speed_matrix/v1_0/t2v_i2v_stage_smoke/summary.json
result/rocm_speed_matrix/aotriton/t2v_i2v_stage_smoke/summary.json
result/rocm_speed_matrix/v1_0/t2v_i2v_stage_smoke/article_t2v_red_cube_256p_8f_s8.mp4
result/rocm_speed_matrix/v1_0/t2v_i2v_stage_smoke/article_i2v_robot_arms_256p_8f_s8.mp4
result/rocm_speed_matrix/aotriton/t2v_i2v_stage_smoke/article_t2v_red_cube_256p_8f_s8.mp4
result/rocm_speed_matrix/aotriton/t2v_i2v_stage_smoke/article_i2v_robot_arms_256p_8f_s8.mp4
```
