# Cosmos3-Nano ROCm v1.1 warm full benchmark 結果

実施日: 2026-06-02

## 目的

事前技術検証で確定した benchmark ルールを runner に実装し、Classmethod 記事条件の T2V / I2V を定常状態で比較する。

採用ルール:

```text
1. pipeline load
2. synthetic VAE warmup
3. 対象 mode の初回 run を warmup として捨てる
4. 2回目を measured benchmark 値として採用
5. VAE mode は default
6. tiling/slicing は無効
```

## 追加実装

`scripts/benchmark_classmethod_article_t2v_i2v_rocm.py` に追加:

```text
--vae-warmup
--vae-warmup-shape
--mode-warmup-runs
--measured-runs
```

`scripts/run_rocm_speed_matrix.py` に追加:

```text
t2v_i2v_article_warm_full
```

dry-run:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --case t2v_i2v_article_warm_full
```

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --case t2v_i2v_article_warm_full \
  --execute
```

## 条件

```text
model: nvidia/Cosmos3-Nano
runtime: PyTorch 2.9.1 + ROCm 7.2
device: AMD Radeon Graphics / gfx1151
height: 256
width: 448
frames: 24
fps: 12
steps: 35
guidance: 1.0
vae warmup shape: [1, 48, 2, 16, 28]
mode warmup runs: 1
measured runs: 1
```

## 結果

### Load / VAE warmup

| Variant | Load | VAE warmup |
| --- | ---: | ---: |
| `v1_0` | 8.229 sec | 370.597 sec |
| `aotriton` | 8.228 sec | 378.286 sec |

VAE warmup は AOTriton では改善しない。

### Measured benchmark

| Case | Variant | Mode warmup run | Measured run | Transformer | VAE decode | Unattributed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T2V | `v1_0` | 111.400 sec | 56.120 sec | 50.768 sec | 4.126 sec | 1.207 sec |
| T2V | `aotriton` | 103.519 sec | 46.925 sec | 41.530 sec | 4.164 sec | 1.211 sec |
| I2V | `v1_0` | 166.851 sec | 167.219 sec | 161.762 sec | 4.168 sec | 1.270 sec |
| I2V | `aotriton` | 94.103 sec | 94.086 sec | 88.623 sec | 4.184 sec | 1.261 sec |

## AOTriton 効果

Measured run の比較:

```text
T2V: 56.120 -> 46.925 sec  (-16.4%)
I2V: 167.219 -> 94.086 sec (-42.7%)
```

Transformer stage の比較:

```text
T2V transformer: 50.768 -> 41.530 sec  (-18.2%)
I2V transformer: 161.762 -> 88.623 sec (-45.2%)
```

AOTriton は VAE decode には効かないが、定常状態の transformer forward には効く。

## Classmethod 記事値との比較

記事の `Article Gen Time` は生成時間であり、動画長ではない。

| Case | Article Gen Time | This ROCm measured | Ratio |
| --- | ---: | ---: | ---: |
| T2V | 22 sec | 46.925 sec | 2.1x slower |
| I2V | 17 sec | 94.086 sec | 5.5x slower |

比較には `aotriton` の measured run を採用した。

以前の cold run 比較:

```text
T2V cold: 483.187 sec -> warm measured aotriton: 46.925 sec
I2V cold: 166.890 sec -> warm measured aotriton: 94.086 sec
```

T2V は cold run では VAE 初回 decode と T2V 初回 pipe 初期化が支配的だったため、定常 benchmark では大きく改善した。

I2V は元から transformer 支配であり、warmup より AOTriton の効果が大きい。

## 判断

full benchmark ルールは有効。

今後の速度比較では、cold start 時間と定常 generation 時間を分けて扱う。

推奨の代表値:

```text
T2V article定常: aotriton measured 46.925 sec
I2V article定常: aotriton measured 94.086 sec
```

出力:

```text
result/rocm_speed_matrix/v1_0/t2v_i2v_article_warm_full/summary.json
result/rocm_speed_matrix/aotriton/t2v_i2v_article_warm_full/summary.json
result/rocm_speed_matrix/v1_0/t2v_i2v_article_warm_full/article_t2v_red_cube_256p_24f_s35.mp4
result/rocm_speed_matrix/v1_0/t2v_i2v_article_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4
result/rocm_speed_matrix/aotriton/t2v_i2v_article_warm_full/article_t2v_red_cube_256p_24f_s35.mp4
result/rocm_speed_matrix/aotriton/t2v_i2v_article_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4
```
