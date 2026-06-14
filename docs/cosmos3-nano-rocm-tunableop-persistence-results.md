# Cosmos3-Nano ROCm TunableOp persistence results

Date: 2026-06-04

## Summary

TunableOp persistence is now wired into the runner and verified.

The earlier missing-file check looked for:

```text
result/rocm_speed_matrix/tunableop_results.csv
```

However, PyTorch TunableOp inserts the device ordinal into the filename. On this single-GPU system, the actual persisted file is:

```text
result/rocm_speed_matrix/tunableop_results0.csv
```

This file exists and has 47 lines.

## Runner changes

Updated `scripts/run_rocm_speed_matrix.py`:

| Variant | Purpose | TunableOp tuning | Filename |
|---|---|---:|---|
| `aotriton_tunable` | Tune and persist results | enabled | `/workspace/result/rocm_speed_matrix/tunableop_results%d.csv` |
| `aotriton_tuned` | Reuse persisted table | disabled | `/workspace/result/rocm_speed_matrix/tunableop_results%d.csv` |

The `%d` is replaced by the GPU ordinal, so GPU 0 writes/reads:

```text
/workspace/result/rocm_speed_matrix/tunableop_results0.csv
```

## Verification

Executed:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case t2v_article_warm_full \
  --execute

python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case i2v_article_warm_full \
  --execute
```

`aotriton_tuned` uses:

```text
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/tunableop_results%d.csv
```

Therefore these runs did not perform online tuning. They loaded the persisted table.

## Results

| Mode | Variant | Total | Transformer | VAE decode |
|---|---|---:|---:|---:|
| T2V | `aotriton_tuned` | 32.165 sec | 26.794 sec | 4.151 sec |
| I2V | `aotriton_tuned` | 77.160 sec | 71.691 sec | 4.195 sec |

Comparison against previous tuning-enabled runs:

| Mode | `aotriton_tunable` transformer | `aotriton_tuned` transformer | Result |
|---|---:|---:|---|
| T2V | 27.031 sec | 26.794 sec | same speed class |
| I2V | 71.643 sec | 71.691 sec | same speed class |

This confirms that the persisted tuning table is being used correctly.

## Operational workflow

Initial or after environment/model change:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tunable \
  --case t2v_article_warm_full \
  --case i2v_article_warm_full \
  --execute
```

Normal reuse without retuning:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case t2v_article_warm_full \
  --case i2v_article_warm_full \
  --execute
```

For WebUI/server operation, use the `aotriton_tuned` environment after the table has been created. Retune only when PyTorch, ROCm, GPU arch, model, dtype, or generation shapes change.

## Outputs

- `result/rocm_speed_matrix/tunableop_results0.csv`
- `result/rocm_speed_matrix/aotriton_tuned/t2v_article_warm_full/summary.json`
- `result/rocm_speed_matrix/aotriton_tuned/i2v_article_warm_full/summary.json`
