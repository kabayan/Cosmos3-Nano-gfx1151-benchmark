# Cosmos3-Nano ROCm v2 hardware bottleneck 調査結果

実施日: 2026-06-03

## 目的

EVO-X2 / Ryzen AI Max+ 395 環境で、v2 の遅さがメモリ帯域頭打ちか、GPU compute / kernel / framework 側の問題かを切り分ける。

今回実施した範囲:

```text
1. rocm-smi logging付きで T2I/T2V/I2V measured を再実行
2. rocm-smi logging付きで Policy normal / skip vision decode を再実行
3. 結果を見て rocprof 対象を絞る
```

## 追加実装

`scripts/run_rocm_speed_matrix.py` に rocm-smi monitor を追加した。

追加オプション:

```text
--rocm-smi-log-dir
--rocm-smi-interval
```

実行例:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case t2i_article_warm_full \
  --case t2v_i2v_article_warm_full \
  --rocm-smi-log-dir result/hwmon_v2 \
  --rocm-smi-interval 5 \
  --execute

python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case policy_article \
  --case policy_article_skip_vision_decode \
  --rocm-smi-log-dir result/hwmon_v2 \
  --rocm-smi-interval 5 \
  --execute
```

## rocm-smi 集計

`rocm-smi` は `Memory Activity` が `N/A` で、実効メモリ帯域は直接取得できなかった。したがって今回の判断は GPU util / clock / power の状況からの一次判断。

| Case | Samples | GPU util p50 | Power p50 | SCLK p50 | MCLK p50 | VRAM p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `t2i_article_warm_full` | 199 | 100% | 112.09 W | 2893 MHz | 1000 MHz | 14% |
| `t2v_i2v_article_warm_full` | 148 | 100% | 110.03 W | 2805 MHz | 1000 MHz | 14% |
| `policy_article` | 384 | 100% | 110.02 W | 2824 MHz | 1000 MHz | 14% |
| `policy_article_skip_vision_decode` | 163 | 100% | 110.02 W | 2812 MHz | 1000 MHz | 14% |

出力:

```text
result/hwmon_v2/aotriton/t2i_article_warm_full/rocm_smi.jsonl
result/hwmon_v2/aotriton/t2v_i2v_article_warm_full/rocm_smi.jsonl
result/hwmon_v2/aotriton/policy_article/rocm_smi.jsonl
result/hwmon_v2/aotriton/policy_article_skip_vision_decode/rocm_smi.jsonl
result/hwmon_v2/aotriton/summary.json
```

## 観察

全ケースで以下の傾向が共通。

```text
GPU util: p50 100%
SCLK: 約2.8-2.9GHz
MCLK: 1000MHzで固定
Power: p50 約110W
VRAM allocated: 14-15%
```

このため、CPU待ちや明確な idle が支配しているとは考えにくい。GPU はほぼ常時動いている。

一方で、rocm-smi ではメモリ帯域 counter が取れていないため、メモリ帯域頭打ちか compute-bound かはまだ断定できない。

## stage 結果との照合

v2 measured stage:

```text
T2I aotriton:
  total 104.297 sec
  transformer 101.961 sec
  VAE decode 1.755 sec

T2V aotriton:
  total 46.996 sec
  transformer 41.599 sec
  VAE decode 4.160 sec

I2V aotriton:
  total 94.219 sec
  transformer 88.753 sec
  VAE decode 4.149 sec
```

Policy benchmark:

```text
Policy video + action:
  generate_batch 1908.169 sec
  generate_samples_from_batch 774.543 sec
  decode 1061.495 sec

Policy skip vision decode:
  generate_batch 786.247 sec
  generate_samples_from_batch 786.112 sec
  decode 0.000 sec
```

## 判断

### T2I/T2V/I2V

T2I/T2V/I2V は GPU util 100%、SCLK 高、MCLK 固定、stage は transformer 支配。

一次判断:

```text
GPU側の transformer kernel が支配。
memory bandwidth-bound の可能性はあるが、rocm-smiだけでは断定不可。
rocprof で matrix/attention kernel の実効帯域と演算効率を見る必要がある。
```

### Policy normal

Policy normal は GPU util 100% で、framework benchmark では vision decode が 1061.495 sec。

一次判断:

```text
Policy video + action の最優先ボトルネックは vision decode。
GPUは使われているため、CPU idleではない。
VAE decode kernel が memory-bound か kernel効率問題かを rocprof で確認する。
```

### Policy skip vision decode

vision decode を外しても GPU util は高く、`generate_samples_from_batch` が約786 sec 残る。

一次判断:

```text
action-onlyでも generate_samples_from_batch の前処理/packing/condition preparation が支配。
sampling progress bar自体は約53 secなので、その前のGPU処理または同期が大きい。
```

## rocprof 対象の絞り込み

優先順:

1. `policy_article` の `OmniMoTModel.decode`
   - 最大ボトルネック: 1061.495 sec
   - 目的: VAE decode が memory-bound か kernel/launch問題か確認。

2. `policy_article_skip_vision_decode` の `generate_samples_from_batch`
   - decode除外後も 786.112 sec。
   - 目的: sampler開始前の前処理/packing/encode/sync の支配kernelを特定。

3. `t2i_article_warm_full` measured transformer
   - 101.961 sec、97%超が transformer。
   - 目的: transformer が memory-bound か compute-bound か確認。

4. `t2v_i2v_article_warm_full` の I2V measured transformer
   - 88.753 sec、I2Vの最大stage。
   - 目的: T2Iと同じkernel傾向か比較。

## 結論

EVO-X2 / Ryzen AI Max+ 395 のメモリ帯域頭打ちの可能性は残る。

ただし今回の rocm-smi では以下までしか言えない。

```text
GPUはほぼ常時100%稼働。
MCLKは1000MHzに張り付き。
Powerは約110W。
CPU待ちやidle支配ではなさそう。
memory bandwidth-boundの断定にはrocprofが必要。
```

次は rocprof で `Policy decode` と `T2I transformer` を優先して取得する。
