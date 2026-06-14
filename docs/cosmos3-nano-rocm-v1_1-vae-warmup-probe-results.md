# Cosmos3-Nano ROCm v1.1 VAE warmup probe 結果

実施日: 2026-06-02

## 目的

VAE decode 単体 probe で、T2V の `vae_decode ~368 sec` が VAE decode 初回コストであることを確認した。

この probe では full benchmark 前の技術検証として、次の 3 点を確認する。

1. pipeline load 後に VAE decode warmup を実行すると、その後の T2V 本体から初回 VAE コストを外せるか。
2. 捕捉済み decode 入力 shape と同じ synthetic latent で warmup が成立するか。
3. `enable_tiling()` / `enable_slicing()` が warmup または本体 T2V に有効か。

## 追加実装

追加スクリプト:

```text
scripts/probe_cosmos3_vae_warmup_rocm.py
```

追加 runner case:

```text
vae_warmup_default
vae_warmup_tiling
vae_warmup_slicing
```

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --case vae_warmup_default \
  --case vae_warmup_tiling \
  --case vae_warmup_slicing \
  --execute
```

## 条件

```text
model: nvidia/Cosmos3-Nano
runtime: PyTorch 2.9.1 + ROCm 7.2
device: AMD Radeon Graphics / gfx1151
height: 256
width: 448
frames: 8
fps: 8
steps: 8
guidance: 1.0
seed: 204
synthetic latent shape: [1, 48, 2, 16, 28]
synthetic latent dtype: torch.float16
```

synthetic latent shape は、VAE decode 単体 probe で T2V pipeline から捕捉した decode 入力と同じ。

## 結果

| Case | VAE mode | Warmup decode | T2V total | T2V transformer | T2V VAE decode | T2V unattributed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vae_warmup_default` | default | 372.357 sec | 62.183 sec | 5.290 sec | 1.031 sec | 55.847 sec |
| `vae_warmup_tiling` | `enable_tiling()` | 334.520 sec | 71.330 sec | 5.342 sec | 1.774 sec | 64.208 sec |
| `vae_warmup_slicing` | `enable_slicing()` | 378.544 sec | 62.711 sec | 5.346 sec | 1.039 sec | 56.321 sec |

比較対象の stage profile:

| Case | Variant | T2V total | T2V VAE decode |
| --- | --- | ---: | ---: |
| no warmup | `v1_0` | 433.401 sec | 368.638 sec |
| no warmup | `aotriton` | 433.241 sec | 368.400 sec |

## 判断

### 1. VAE warmup は有効

default warmup により、T2V total は 433.401 sec から 62.183 sec へ短縮した。

```text
T2V total:      433.401 -> 62.183 sec
T2V vae_decode: 368.638 ->  1.031 sec
```

VAE decode 初回コストを pipeline load 後の warmup に移せば、T2V 本体の計測からは外せる。

### 2. synthetic latent warmup は成立

pipeline から捕捉した shape `[1, 48, 2, 16, 28]` と同じ synthetic tensor で warmup できた。

これにより、実生成を一度走らせて latent を捕捉しなくても、load 後に固定 shape の synthetic warmup を実行できる。

### 3. tiling/slicing は優先しない

`enable_tiling()` は warmup decode 自体を 372.357 sec から 334.520 sec に短縮したが、その後の T2V total は 62.183 sec から 71.330 sec に悪化した。T2V 内の VAE decode も 1.031 sec から 1.774 sec に悪化した。

`enable_slicing()` は default とほぼ同等で、明確な改善はない。

したがって、現時点では full benchmark の推奨設定に tiling/slicing は入れない。

## 残課題

warmup 後も T2V total には約 56 秒の unattributed pipe time が残る。

```text
default warmup:
  transformer_forward: 5.290 sec
  vae_decode: 1.031 sec
  video_postprocess: 0.015 sec
  unattributed_pipe: 55.847 sec
```

full benchmark 前の追加検証としては、広い探索ではなく、この unattributed pipe time の内訳確認だけを行う価値がある。

候補:

- prompt encode / text path の stage profile 追加。
- scheduler / latent preparation / safety checker 周辺の stage profile 追加。
- warmup 後 T2V を同一プロセスで 2 回連続実行し、unattributed が初回 T2V 固有か定常コストか確認。

## T2V 2連続 run probe

追加で、同一プロセス内で synthetic VAE warmup 後に T2V を 2 回連続実行した。

追加 runner case:

```text
vae_warmup_t2v_twice
```

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --case vae_warmup_t2v_twice \
  --execute
```

結果:

| Step | Seconds | Transformer | VAE decode | Postprocess | Unattributed |
| --- | ---: | ---: | ---: | ---: | ---: |
| VAE warmup | 375.461 sec | - | 375.461 sec | - | - |
| T2V run 1 | 62.519 sec | 5.303 sec | 1.031 sec | 0.005 sec | 56.180 sec |
| T2V run 2 | 6.245 sec | 4.928 sec | 1.013 sec | 0.005 sec | 0.299 sec |

判断:

- VAE warmup 後に残っていた `unattributed_pipe ~56 sec` は、T2V 初回 run 固有の追加初期化コストだった。
- 同一プロセス内の T2V 2 回目は、ほぼ transformer + VAE decode + postprocess の実処理時間まで落ちる。
- T2V の定常 smoke 性能は 8 steps / 8 frames 条件で約 6.2 sec。

この結果により、full benchmark では「VAE warmup だけ」ではなく、「計測対象と同じモードの初回 run を捨てる」必要がある。

## full benchmark への反映方針

推奨:

```text
VAE mode: default
VAE warmup: enabled
warmup latent shape: [1, 48, 2, 16, 28]
mode warmup: measured mode の initial run を1回捨てる
tiling: disabled
slicing: disabled
```

T2V/I2V article benchmark は、まず default VAE warmup ありで実施し、各 mode の初回 run を warmup として扱う。速度比較に使う値は同一プロセス内の 2 回目以降を優先する。

出力:

```text
result/rocm_speed_matrix/v1_0/vae_warmup_default/vae_warmup_probe.json
result/rocm_speed_matrix/v1_0/vae_warmup_tiling/vae_warmup_probe.json
result/rocm_speed_matrix/v1_0/vae_warmup_slicing/vae_warmup_probe.json
result/rocm_speed_matrix/v1_0/vae_warmup_t2v_twice/vae_warmup_probe.json
```
