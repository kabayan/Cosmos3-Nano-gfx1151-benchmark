# PyTorch 2.13 + AOTriton 0.12b 隔離 T2V 検証

- 日付: 2026-07-28
- 実行者: Codex（`/dls-continue` で DLS-020 限定検証 (2) を執行）
- 目的: 現行 ROCm 7.2 userspace と Diffusers 実装を固定し、PyTorch 2.9.1 から公式 2.13.0 ROCm 7.2 wheel への更新を同一 T2V プロトコルで比較する

## 候補と隔離方式

- 採用: 現行 `cosmos3-rocm72-diffusers:local` から派生する再現可能な Docker image。公式 ROCm 7.2 index の `torch==2.13.0` / `torchvision==0.28.0` だけを差し替える
- 不採用: 現行 image 内で直接 pip upgrade（既存 2.9.1 環境を破壊する）
- 不採用: 一時 container を `docker commit` だけで保存（依存解決が Dockerfile に残らず再現不能）

実装:

- `docker/cosmos3-rocm72-diffusers-torch213.Dockerfile`
- image: `cosmos3-rocm72-diffusers:torch213`
- image ID: `sha256:48ff721a2c79bfd6904705f1d0ee0e4b09441dbd0b05a9cf54ae3bc68b73a2d7`

公式 index の wheel 一覧は 2026-07-28 に次を確認した。

- `torch 2.13.0+rocm7.2` cp312
- `torchvision 0.28.0+rocm7.2` cp312
- `torchaudio` は 2.11.0 までで 2.13.0 が無い

参照:

- https://download.pytorch.org/whl/rocm7.2/torch/
- https://download.pytorch.org/whl/rocm7.2/
- https://github.com/pytorch/pytorch/releases/tag/v2.13.0

## セットアップ中に判明した互換性問題

base image の `torchaudio 2.9.0+rocm7.2` を残すと、Transformers の optional import が `libtorchaudio.so` を読み、torch 2.13 ABI に存在しない symbol で Diffusers import が失敗した。

```text
OSError: libtorchaudio.so: undefined symbol: ... torch::Library::_def ...
```

公式 ROCm 7.2 index に torchaudio 2.13 wheel が無く、T2V は torchaudio を使わないため、派生 image から旧 torchaudio を削除した。修正後の GPU smoke:

```text
torch 2.13.0+rocm7.2 hip 7.2.53211
torchvision 0.28.0+rocm7.2
triton-rocm 3.7.1
gpu AMD Radeon 8060S gfx1151
cache_api True
flash_available True
aotriton_libs ['aotriton.images', 'libaotriton_v2.so', 'libaotriton_v2.so.0.12.0']
sdpa (1, 8, 1024, 128) torch.float16 True
```

## TunableOp 表の互換性

現行 2.9.1 表の validator は PT 2.9.1、hipBLASLt `100201-5b515cf1bc`、rocBLAS `5.2.0.5b515cf1bc`。2.13 runtime は PT 2.13.0、hipBLASLt `100202-dabb6df2b9`、rocBLAS `5.2.0.dabb6df2b9`。

2.13 は旧表を `Failed validator: ROCBLAS_VERSION` として拒否した。validator を書き換えて旧 solver ID を強制するのはライブラリ build が異なり不正なため行わなかった。

## 同一プロトコル run

成果物:

- `result/t2v_und_cache_torch213_official_20260728/summary.json`
- `result/t2v_und_cache_torch213_official_20260728/run.log`
- `result/t2v_und_cache_torch213_official_20260728/article_t2v_red_cube_256p_24f_s35.mp4`

DLS-021 の baseline と同じ guidance 6.0、35 steps、256x448、24 requested frames、seed 202、AOTriton env、und branch cache、VAE warmup、mode warmup 1 + measured 1 を使用した。旧 TunableOp 表は runtime が拒否したため 2.13 default solver の互換性対照となった。

| 指標 | 2.9.1 + 有効な調律表 | 2.13 default | 差 |
|---|---:|---:|---:|
| measured total | 40.806 秒 | 53.361 秒 | +30.77% |
| transformer | 35.552 秒 | 48.100 秒 | +35.30% |
| VAE decode | 4.028 秒 | 4.057 秒 | ほぼ不変 |
| cache stats | 2 writes / 138 reads | 2 writes / 138 reads | 同一 |

## 出力非等価

- 2.9.1 baseline MP4 SHA-256: `d086071dc8808359cf743322233910b510dbee9e0332cd4c5f224f4ca9908373`
- 2.13 MP4 SHA-256: `b68881e423373a9961932fe01f7fbb04650914ce7113ebe5620c7dd91c1951b7`
- `cmp -s`: exit 1
- ffmpeg decoded framemd5: 全 21 frame が相違
- decoded video PSNR: 28.411745 dB
- decoded video SSIM: 0.948815
- codec / resolution / frame count / fps / pixel format は同一（H.264, 448x256, 21 frames, 12 fps, yuv420p）

したがって差は container や MP4 metadata だけではなく生成画素に存在する。

## 判定

- 不採用。DLS-020 が diffusers 経路に要求する「同一入力・seed の出力 hash 一致」を満たさない
- 速度は未調律のため 2.13 自体の最終性能値ではないが、出力不一致により現構成は採用資格を失った。再調律で現行 hash を回復する根拠はなく、限定検証を solver 探索 campaign へ拡張しない
- 現行 stable stack（PyTorch 2.9.1 + 検証済み TunableOp 表）を維持する
- AOTriton PR #203/#205 完了後の head_dim=128 probe は別候補として残す
