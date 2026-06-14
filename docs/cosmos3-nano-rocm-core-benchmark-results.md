# Cosmos3-Nano ROCm Core Benchmark Results

実施日: 2026-06-01

対象: `nvidia/Cosmos3-Nano`

## 結論

Core suite は全ケース 3 回ずつ成功した。

重要な観察:

- 初回 generation だけ非常に遅く、2回目以降は大幅に速い。
- `t2i_480_fp16_s4_g1` と `t2i_480_bf16_s4_g1` は初回が約 327-329 sec、2回目以降は約 3.5 sec。
- `t2v5_256_fp16_s4_g1` は初回が 338.849 sec、2回目以降は 3.761-4.366 sec。
- この差は VAE decode / postprocess / kernel 初期化 / cache warm-up の影響が大きい可能性が高い。
- `t2i_480_fp16_s8_g1` は直前ケースで warm-up 済みだったため、3回とも 5.7-6.0 sec と安定した。

## Summary

| Case | Runs | Passed | Mean sec | Min sec | Max sec | Stdev sec | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `load_fp16` | 3 | 3 | 7.879 | 6.953 | 8.833 | 0.940 | 11.93 |
| `t2i_256_fp16_s4_g1` | 3 | 3 | 33.527 | 1.883 | 96.815 | 54.809 | 163.48 |
| `t2i_480_fp16_s4_g1` | 3 | 3 | 111.263 | 3.500 | 326.762 | 186.628 | 167.74 |
| `t2i_480_bf16_s4_g1` | 3 | 3 | 112.113 | 3.504 | 329.321 | 188.108 | 167.78 |
| `t2i_480_fp16_s8_g1` | 3 | 3 | 5.831 | 5.731 | 6.004 | 0.150 | 2.58 |
| `t2v5_256_fp16_s4_g1` | 3 | 3 | 115.659 | 3.761 | 338.849 | 193.289 | 167.12 |

## Run Files

出力先:

```text
result/benchmark/core/
```

主要ファイル:

| File | 内容 |
| --- | --- |
| `summary.csv` | ケース別集計 |
| `summary.json` | ケース別集計 JSON |
| `runs.csv` | run 別詳細 |
| `runs.jsonl` | run 別詳細 JSONL |
| `benchmark.log` | 実行ログ |
| `*.jpg` | image benchmark 出力 |
| `*.mp4` | video benchmark 出力 |

## Notes

- 測定値は pipeline をケースごとに 1 回ロードし、その後 generation を 3 回実行したもの。
- `load_seconds_once` は `runs.csv` 側に記録済み。
- 初回と2回目以降の差が大きいため、今後は cold run と warm run を分けて集計する方がよい。
- `HF_HUB_DISABLE_XET=1` を設定して実行した。
