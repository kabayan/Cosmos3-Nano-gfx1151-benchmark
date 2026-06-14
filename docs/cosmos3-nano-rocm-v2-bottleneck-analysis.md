# Cosmos3-Nano ROCm v2 ボトルネック分析

実施日: 2026-06-02

## v2 の定義

この時点の v2 は以下を含む。

```text
T2I/T2V/I2V:
  AOTriton 有効
  synthetic VAE warmup
  mode 初回 run を warmup として捨てる
  2回目を measured 値として採用

Policy:
  cosmos-framework 経路
  AOTriton 有効
  benchmark.json による framework timer 取得
  vision decode skip probe による action-only 改善上限確認
```

## v2 measured 結果

| Mode | Variant | Total | Transformer | VAE decode | Unattributed |
| --- | --- | ---: | ---: | ---: | ---: |
| T2I | `aotriton` | 104.733 sec | 102.378 sec | 1.769 sec | 0.578 sec |
| T2V | `aotriton` | 46.925 sec | 41.530 sec | 4.164 sec | 1.211 sec |
| I2V | `aotriton` | 94.086 sec | 88.623 sec | 4.184 sec | 1.261 sec |

stage 比率:

```text
T2I transformer: 97.8%
T2V transformer: 88.5%
I2V transformer: 94.2%
```

T2I/T2V/I2V の v2 定常状態では、主ボトルネックは transformer forward。

VAE decode は warmup 後には小さい。

```text
T2I VAE decode: 1.769 sec
T2V VAE decode: 4.164 sec
I2V VAE decode: 4.184 sec
```

## cold start ボトルネック

VAE decode 初回は依然として非常に重い。

```text
v1_0 standalone VAE decode run 1: 372.683 sec
v1_0 standalone VAE decode run 2:   1.019 sec
aotriton standalone VAE decode run 1: 374.573 sec
aotriton standalone VAE decode run 2:   1.023 sec
```

T2I 960x960 用 warmup ではさらに大きい。

```text
T2I 960x960 VAE warmup v1_0: 661.479 sec
T2I 960x960 VAE warmup aotriton: 677.865 sec
```

判断:

- cold start では VAE 初回 decode が最大級のボトルネック。
- AOTriton は VAE 初回 decode には効かない。
- 常駐プロセス + 起動時 warmup が必須。

## T2I/T2V/I2V のボトルネック

### T2I

```text
total: 104.733 sec
transformer: 102.378 sec
vae_decode: 1.769 sec
```

T2I は transformer が 97.8% を占める。960x960 / 35 steps の解像度・step 数が直接効いている。

改善候補:

- AOTriton は採用済み。
- step 数削減。
- 解像度削減。
- transformer backend / attention backend の追加最適化。
- TunableOp は transformer matmul に効く可能性があるが、以前の collect では CSV 未生成で未確定。

### T2V

```text
total: 46.925 sec
transformer: 41.530 sec
vae_decode: 4.164 sec
```

T2V は transformer が 88.5%。VAE decode は 8.9% まで下がった。

改善候補:

- step 数削減。
- frame 数削減。
- AOTriton は採用済み。
- transformer 追加最適化。

### I2V

```text
total: 94.086 sec
transformer: 88.623 sec
vae_decode: 4.184 sec
```

I2V は transformer が 94.2%。AOTriton の効果が最も大きいが、まだ transformer 支配。

改善候補:

- step 数削減。
- frame 数削減。
- transformer 追加最適化。
- 入力画像条件処理の stage は現状では支配的ではない。

## Policy のボトルネック

Policy は T2I/T2V/I2V と異なり、cosmos-framework 経路。

通常の video + action 出力:

```text
OmniInference.generate_batch: 1908.169 sec
OmniMoTModel.generate_samples_from_batch: 774.543 sec
OmniMoTModel.decode: 1061.495 sec
```

vision decode skip probe:

```text
OmniInference.generate_batch: 786.247 sec
OmniMoTModel.generate_samples_from_batch: 786.112 sec
OmniMoTModel.decode: 0.000 sec
```

判断:

- Policy video + action 出力の最大ボトルネックは `OmniMoTModel.decode`。
- decode だけで 1061.495 sec。
- vision decode を外すと 1908.169 sec から 786.247 sec に短縮する。
- action-only 用途なら正式に vision decode/export を skip する mode が最も効果的。
- 記事同等の video + action を維持する場合、Wan VAE / vision decode の高速化が必要。

`generate_samples_from_batch` も 774-786 sec あり、sampling progress bar 約 52 sec よりはるかに長い。sampling 前の latent/condition preparation、packing、encode、または同期が残りの大きなボトルネック。

## v2 ボトルネックまとめ

| Area | Current bottleneck | Severity | Next action |
| --- | --- | --- | --- |
| Cold start | VAE first decode | High | 常駐化 + warmup 維持 |
| T2I | transformer forward | High | step/resolution 削減、backend 最適化 |
| T2V | transformer forward | Medium | step/frame 削減、backend 最適化 |
| I2V | transformer forward | High | step/frame 削減、backend 最適化 |
| Policy video + action | vision decode | Critical | vision decode warmup/chunk/backend 調査 |
| Policy action-only | generate_samples前処理 | High | framework 内部 stage 分解 |

## 推奨する次の改善順

1. Policy action-only 正式 mode を実装する。
   - Action だけ必要な用途では最大効果。
   - 現 probe では 2.4x faster の上限を確認済み。

2. Policy `generate_samples_from_batch` 内部をさらに分解する。
   - sampling progress bar は約52 sec。
   - しかし method 全体は約786 sec。
   - sampling 前処理が大きすぎるため、encode / packing / condition preparation / sync を分解する。

3. Policy video decode の warmup / chunk / backend を調査する。
   - 記事同等 video + action を維持するなら必須。

4. T2I/T2V/I2V は transformer 最適化へ進む。
   - AOTriton は採用済み。
   - 次は TunableOp 再検証、step/frame/resolution tradeoff、または torch.compile/cuda graphs 相当の ROCm 可否確認。

## 出力

```text
result/rocm_speed_matrix/aotriton/t2i_article_warm_full/article_t2i_summary.json
result/rocm_speed_matrix/aotriton/t2v_i2v_article_warm_full/summary.json
result/rocm_speed_matrix/aotriton/policy_article/benchmark.json
result/rocm_speed_matrix/aotriton/policy_article_skip_vision_decode/benchmark.json
```
