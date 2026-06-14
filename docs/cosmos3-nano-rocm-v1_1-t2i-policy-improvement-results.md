# Cosmos3-Nano ROCm v1.1 T2I / Policy 改善計画と実施結果

実施日: 2026-06-02

## 目的

T2V / I2V で有効だった改善ルールを T2I と Policy Model にも適用または検証し、v1 cold run からの改善余地を確認する。

## 改善計画

### Text-to-image

T2I は diffusers `Cosmos3OmniPipeline` 経路なので、T2V / I2V と同じ rule を適用する。

```text
1. pipeline load
2. 960x960 用 synthetic VAE warmup
3. T2I 初回 run を warmup として捨てる
4. 2回目を measured benchmark 値として採用
5. AOTriton variant と比較
6. stage profile で transformer / VAE decode / postprocess を分解
```

T2I warmup latent shape:

```text
[1, 48, 1, 60, 60]
```

### Policy Model

Policy Model は cosmos-framework 経路で、diffusers の warm measured rule をそのまま適用できない。

まず AOTriton variant を実行し、v1 end-to-end と比較する。

```text
1. existing policy_article を aotriton variant で実行
2. console log の時刻から end-to-end を比較
3. sampling と sampling 前後の支配関係を確認
4. 改善が見えない場合は framework 内 stage profile を次フェーズ候補にする
```

## 追加実装

`scripts/benchmark_classmethod_article_t2i_rocm.py` に追加:

```text
--stage-profile
--vae-warmup
--vae-warmup-shape
--mode-warmup-runs
--measured-runs
```

`scripts/run_rocm_speed_matrix.py` に追加:

```text
t2i_article_warm_full
```

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant v1_0 \
  --variant aotriton \
  --case t2i_article_warm_full \
  --execute

python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case policy_article \
  --execute
```

## T2I 結果

条件:

```text
height: 960
width: 960
frames: 1
steps: 35
guidance: 1.0
```

| Variant | VAE warmup | Mode warmup run | Measured run | Transformer | VAE decode | Unattributed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_0` | 661.479 sec | 317.099 sec | 206.367 sec | 203.997 sec | 1.774 sec | 0.588 sec |
| `aotriton` | 677.865 sec | 226.267 sec | 104.733 sec | 102.378 sec | 1.769 sec | 0.578 sec |

v1 cold run との比較:

```text
T2I cold v1: 974.611 sec
T2I warm measured v1_0: 206.367 sec
T2I warm measured aotriton: 104.733 sec
```

改善:

```text
cold v1 -> warm measured aotriton: 9.3x faster
warm v1_0 -> warm aotriton: 49.3% faster
```

記事値との比較:

```text
Article Gen Time: 22 sec
ROCm warm measured aotriton: 104.733 sec
Ratio: 4.8x slower
```

判断:

- T2I も VAE 初回 decode と mode 初回 run を捨てることで大きく改善した。
- 定常 T2I は transformer 支配。
- AOTriton は T2I transformer に明確に効く。
- VAE warmup 自体は AOTriton では改善しない。

## Policy Model 結果

実行:

```text
variant: aotriton
case: policy_article
```

出力:

```text
result/rocm_speed_matrix/aotriton/policy_article/action_policy_robot/sample_outputs.json
result/rocm_speed_matrix/aotriton/policy_article/console.log
```

console log 時刻:

```text
Loaded samples: 20:50:46
Saved sample args: 20:51:12
Sampling start: 21:03:37
Saved sample outputs: 21:23:24
```

概算:

```text
end-to-end: 1958 sec
sample args saved -> sampling start: 745 sec
sampling start -> output saved: 1187 sec
sampling progress bar: about 52 sec
```

v1 との比較:

```text
Policy cold v1: ~1965 sec
Policy aotriton: ~1958 sec
```

判断:

- Policy Model は AOTriton だけでは end-to-end の改善がほぼない。
- sampling progress bar は約52秒で完了しているが、sampling 前後の framework 処理が支配的。
- 次に改善するなら、cosmos-framework 内の stage profile が必要。
- 現時点では Policy の改善は未達。T2I/T2V/I2V と同じ diffusers warm measured rule では扱えない。

## Policy stage profile / vision decode skip probe

Policy wrapper に `--benchmark` を追加し、framework の timer を `benchmark.json` として出力するようにした。

追加実装:

```text
scripts/run_cosmos_framework_policy_rocm.py --skip-vision-decode
scripts/run_rocm_speed_matrix.py case: policy_article_skip_vision_decode
```

`--skip-vision-decode` は `OmniMoTModel.decode()` をゼロ tensor 返却に置き換える probe。Action 出力は維持するが、vision.mp4 は実映像ではなくゼロ映像になる。したがって記事の Policy video 出力との比較値ではなく、action-only 相当の改善上限確認として扱う。

実行:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case policy_article \
  --execute

python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case policy_article_skip_vision_decode \
  --execute
```

benchmark 結果:

| Case | generate_batch | generate_samples_from_batch | decode |
| --- | ---: | ---: | ---: |
| `policy_article` | 1908.169 sec | 774.543 sec | 1061.495 sec |
| `policy_article_skip_vision_decode` | 786.247 sec | 786.112 sec | 0.000 sec |

ログ時刻:

| Case | sample args saved | sampler start | output saved |
| --- | --- | --- | --- |
| `policy_article` | 21:45:10 | 21:57:12 | 22:16:59 |
| `policy_article_skip_vision_decode` | 22:31:56 | 22:44:09 | 22:45:02 |

判断:

- Policy の最大支配 stage は vision decode。`OmniMoTModel.decode` だけで 1061.495 sec。
- vision decode を外すと `generate_batch` は 1908.169 sec から 786.247 sec へ短縮する。
- action tensor はどちらも `16x10` で出力された。
- 残る 786 sec は `generate_samples_from_batch` 支配。sampling progress bar 自体は約 52 sec なので、sampler 開始前の準備がさらに大きい。
- 実用上、Policy を action-only 用途で使うなら vision decode/export を無効化する専用経路が最も効果的。
- 記事同等の video + action 出力を維持する場合は、vision decode を高速化しない限り大幅改善は難しい。

## 総合判断

改善済み:

```text
T2I: 974.611 -> 104.733 sec, 9.3x faster
T2V: 483.187 -> 46.925 sec, 10.3x faster
I2V: 166.890 -> 94.086 sec, 1.8x faster
```

未改善:

```text
Policy video + action: ~1965 -> ~1958 sec, effectively unchanged
Policy action-only probe: ~1958 -> 786 sec, 2.4x faster
```

次の Policy 改善候補:

1. action-only 正式 mode の実装。vision decode と vision.mp4 保存を完全に skip する。
2. `generate_samples_from_batch` 内の sampling 前処理を分解する。
3. framework tokenizer/VAE decode の warmup 可否確認。
4. 同一 process 内で複数 sample を流せる runner 化。
5. video + action 出力が必須の場合は Wan VAE decode の warmup / chunk / backend 調査。

## 出力

```text
result/rocm_speed_matrix/v1_0/t2i_article_warm_full/article_t2i_summary.json
result/rocm_speed_matrix/aotriton/t2i_article_warm_full/article_t2i_summary.json
result/rocm_speed_matrix/aotriton/policy_article/action_policy_robot/sample_outputs.json
result/rocm_speed_matrix/aotriton/policy_article/benchmark.json
result/rocm_speed_matrix/aotriton/policy_article_skip_vision_decode/benchmark.json
result/rocm_speed_matrix/aotriton/policy_article_skip_vision_decode/action_policy_robot/sample_outputs.json
```
