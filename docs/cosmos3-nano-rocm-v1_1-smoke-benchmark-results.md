# Cosmos3-Nano ROCm v1.1 smoke benchmark 結果

実施日: 2026-06-02

## 目的

full benchmark 前に、技術検証を通過した variant だけを短い生成条件で実行し、生成 pipeline と出力ファイルが成立するか確認する。

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --variant tunable_collect \
  --case t2i_smoke \
  --case t2v_i2v_smoke \
  --execute
```

## 条件

### T2I smoke

```text
height: 480
width: 480
frames: 1
steps: 8
guidance: 1.0
seed: 201
```

### T2V/I2V smoke

```text
height: 256
width: 448
frames: 8
fps: 8
steps: 8
guidance: 1.0
T2V seed: 202
I2V seed: 203
```

## 結果

| Variant | T2I | T2V | I2V |
| --- | ---: | ---: | ---: |
| `v1_0` | 225.484 sec | 435.044 sec | 31.803 sec |
| `aotriton` | 210.966 sec | 439.493 sec | 18.800 sec |
| `tunable_collect` | 228.752 sec | 441.751 sec | 31.869 sec |

v1.0 比:

| Variant | T2I | T2V | I2V |
| --- | ---: | ---: | ---: |
| `aotriton` | -6.44% | +1.02% | -40.89% |
| `tunable_collect` | +1.45% | +1.54% | +0.21% |

## 所見

### AOTriton

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` は T2I と I2V で改善した。

```text
T2I: 225.484 sec -> 210.966 sec
I2V: 31.803 sec -> 18.800 sec
```

I2V は denoise が支配的なため、技術検証で確認した fused SDPA backend 有効化の効果が出ている可能性が高い。

一方、T2V は総時間が後処理支配で、AOTriton の denoise 短縮が総時間に反映されていない。

```text
T2V: 435.044 sec -> 439.493 sec
```

### TunableOp collect

`tunable_collect` は tuning table 作成前の収集モードなので、速度改善は期待しない。

結果は v1.0 とほぼ同等または少し遅い。

```text
T2I: +1.45%
T2V: +1.54%
I2V: +0.21%
```

次に `tunableop_untuned.csv` を確認し、offline tuning table を作成する。

確認結果:

```text
result/rocm_speed_matrix/tunableop_untuned.csv: not generated
result/rocm_speed_matrix/tunableop_results.csv: not generated
```

したがって、次の TunableOp 作業では benchmark 再実行前に `torch.cuda.tunable` の API / env name / file output 条件を単体で確認する。

### T2V 後処理

T2V smoke は denoise 8 steps が短時間で完了しても、総時間は約 435-442 sec だった。

これは v1.0 / article benchmark と同じく、T2V は VAE decode / postprocess / export が支配的であることを示している。

## Gate 判定

### article benchmark に進める

- `v1_0`
- `aotriton`

理由:

- 両方とも smoke 出力成功。
- `aotriton` は T2I / I2V に明確な改善がある。

### tuning 後に再評価

- `tunable_collect`

理由:

- collect mode は速度比較用ではない。
- TunableOp table 作成後に `tunable_online` または table 読み込み variant で再測定する。

### full benchmark で優先しない

- T2V smoke のままでは、AOTriton だけでは総時間改善が出ていない。
- T2V は stage timer / decode-export 分解を優先する。

## 出力ファイル

```text
result/rocm_speed_matrix/v1_0/t2i_smoke/article_t2i_summary.json
result/rocm_speed_matrix/v1_0/t2v_i2v_smoke/summary.json
result/rocm_speed_matrix/aotriton/t2i_smoke/article_t2i_summary.json
result/rocm_speed_matrix/aotriton/t2v_i2v_smoke/summary.json
result/rocm_speed_matrix/tunable_collect/t2i_smoke/article_t2i_summary.json
result/rocm_speed_matrix/tunable_collect/t2v_i2v_smoke/summary.json
```

生成 media:

```text
result/rocm_speed_matrix/*/t2i_smoke/article_t2i_robotics_lab_480x480_s8.jpg
result/rocm_speed_matrix/*/t2v_i2v_smoke/article_t2v_red_cube_256p_8f_s8.mp4
result/rocm_speed_matrix/*/t2v_i2v_smoke/article_i2v_robot_arms_256p_8f_s8.mp4
```

## 次の作業

1. TunableOp の file output 条件を単体 probe で確認する。
2. `tunableop_untuned.csv` を生成できる設定に修正する。
3. TunableOp offline tuning table を作成する。
4. T2V の stage timer を追加して、VAE decode / export を分解する。完了: `docs/cosmos3-nano-rocm-v1_1-stage-profile-results.md`
5. `aotriton` で T2V stage profile を再実行し、VAE decode に AOTriton が効くか確認する。完了: T2V VAE decode はほぼ変化なし、I2V transformer は約 43% 短縮。
6. T2V の VAE decode 単体 probe を追加し、latent shape / dtype / chunk 設定 / profiler trace を確認する。
7. `v1_0` と `aotriton` で T2I/I2V article benchmark を実行する。
