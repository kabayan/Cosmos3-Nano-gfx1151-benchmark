# Cosmos3-Nano ROCm v1.1 VAE decode 単体 probe 結果

実施日: 2026-06-02

## 目的

T2V stage profile で `vae_decode` が約 368 秒を占めていたため、pipeline 全体から VAE decode 入力を切り出し、同じ latent を使って VAE decode 単体の初回/2回目を計測する。

確認したい点:

- T2V の長時間化が VAE decode 単体で再現するか。
- 初回だけ遅いのか、定常 decode も遅いのか。
- AOTriton が VAE decode 初回コストに効くか。

## 追加実装

追加スクリプト:

```text
scripts/probe_cosmos3_vae_decode_rocm.py
```

`Cosmos3OmniPipeline` の `pipe.vae.decode` を一時的に monkey patch し、decode に渡される tensor 入力を clone して捕捉する。

`--abort-after-capture` を指定した場合は、decode 入力を捕捉した時点で pipeline を中断し、元の `vae.decode` を使って単体 decode を実行する。これにより、pipeline 内 decode と単体 decode の初回コストを分けて確認できる。

`run_rocm_speed_matrix.py` に追加した case:

```text
vae_decode_probe
```

実行コマンド:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --case vae_decode_probe \
  --execute

python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case vae_decode_probe \
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
standalone decode runs: 2
```

## 捕捉した VAE decode 入力

```text
vae class: AutoencoderKLWan
vae module: diffusers.models.autoencoders.autoencoder_kl_wan
vae dtype: torch.float16
decode input shape: [1, 48, 2, 16, 28]
decode input dtype: torch.float16
decode input device: cuda:0
decode input numel: 43008
```

利用可能な VAE メソッド:

```text
enable_tiling
disable_tiling
enable_slicing
disable_slicing
enable_gradient_checkpointing
```

## 結果

### 事前確認

最初の probe では pipeline を最後まで実行した後、捕捉した同じ latent で単体 decode を実行した。

```text
pipe_seconds: 429.922 sec
standalone_decode run 1: 1.001 sec
```

この結果から、pipeline 完了後の定常 decode は約 1 秒であり、stage profile の `vae_decode ~368 sec` は定常性能ではなく初回コストの可能性が高いと判断した。

### abort-after-capture probe

decode 入力を捕捉した時点で pipeline を中断し、その直後に単体 decode を 2 回実行した。

| Variant | Pipeline status | Decode run 1 | Decode run 2 |
| --- | --- | ---: | ---: |
| `v1_0` | `aborted_after_capture` | 372.683 sec | 1.019 sec |
| `aotriton` | `aborted_after_capture` | 374.573 sec | 1.023 sec |

出力:

```text
result/rocm_speed_matrix/v1_0/vae_decode_probe/vae_decode_probe.json
result/rocm_speed_matrix/aotriton/vae_decode_probe/vae_decode_probe.json
```

## stage profile との照合

stage profile の T2V 結果:

| Variant | T2V total pipe | T2V vae_decode |
| --- | ---: | ---: |
| `v1_0` | 433.401 sec | 368.638 sec |
| `aotriton` | 433.241 sec | 368.400 sec |

VAE decode 単体 probe:

| Variant | Standalone decode run 1 | Standalone decode run 2 |
| --- | ---: | ---: |
| `v1_0` | 372.683 sec | 1.019 sec |
| `aotriton` | 374.573 sec | 1.023 sec |

stage profile の `vae_decode` と単体 probe の `run 1` は同じ 370 秒級であり、T2V の支配要因は VAE decode の初回実行コストとして単体再現できた。

## 結論

- T2V の 368 秒級 `vae_decode` は VAE decode 単体の初回実行で再現した。
- 同じ入力の 2 回目 decode は約 1 秒なので、定常 decode 性能は遅くない。
- AOTriton は VAE decode 初回コストには効かない。
- AOTriton は I2V transformer には効くが、T2V の主問題である VAE decode 初回には効果がない。

## 次の検証候補

T2V 高速化の次フェーズでは、VAE decode 初回コストを本番計測から外せるかを確認する。

優先候補:

1. VAE warmup を pipeline ロード直後に実行し、その後の T2V が 433 秒級から短縮するか確認する。
2. warmup の latent shape を `[1, 48, 2, 16, 28]` に固定した synthetic tensor で再現できるか確認する。
3. `enable_tiling()` / `enable_slicing()` の有無で初回 decode と定常 decode が変わるか確認する。
4. profiler trace を取得し、初回 370 秒の内訳が kernel compile、MIOpen find、attention backend 初期化、その他のどれかを特定する。

full benchmark 前には、まず VAE warmup smoke を追加して T2V article benchmark の短縮効果を確認する。
