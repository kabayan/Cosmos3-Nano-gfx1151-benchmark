# MIOpen large BF16 Conv3d solver request

## Environment

- GPU: gfx1151 / AMD Radeon Graphics
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- MIOpen: `3.5.1.5b515cf1bc`
- Find mode: `MIOPEN_FIND_MODE=NORMAL`
- Find enforce: `MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE`

## Problem

Cosmos3 Policy video decode hits large BF16 3D convolution descriptors that MIOpen dispatches to `ConvDirectNaiveConvFwd` with workspace `0`.

These descriptors dominate decode time. MIOpen FindDb tuning and warmed user FindDb do not improve the full Policy workload.

## Repro commands

Descriptor 1:

```bash
./bin/MIOpenDriver convbfp16 \
  -n 1 -c 160 --in_d 18 -H 274 -W 370 \
  -k 160 --fil_d 3 -y 3 -x 3 \
  --pad_d 0 -p 0 -q 0 \
  --conv_stride_d 1 -u 1 -v 1 \
  --dilation_d 1 -l 1 -j 1 \
  --spatial_dim 3 -m conv -g 1 -F 1 -t 1
```

Observed:

- Descriptor: `160-18-274-370-3x3x3-160-16-272-368-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F`
- `GetWorkSpaceSize`: `0`
- Final solver: `ConvDirectNaiveConvFwd`
- Estimated solver time in Find log: about `19855-19870 ms`
- Direct PyTorch full cold execution: `180.883 sec`
- Direct PyTorch same-process warmed execution: `19.985 sec`

Descriptor 2:

```bash
./bin/MIOpenDriver convbfp16 \
  -n 1 -c 512 --in_d 6 -H 242 -W 322 \
  -k 256 --fil_d 3 -y 3 -x 3 \
  --pad_d 0 -p 0 -q 0 \
  --conv_stride_d 1 -u 1 -v 1 \
  --dilation_d 1 -l 1 -j 1 \
  --spatial_dim 3 -m conv -g 1 -F 1 -t 1
```

Observed:

- Descriptor: `512-6-242-322-3x3x3-256-4-240-320-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F`
- `GetWorkSpaceSize`: `0`
- Final solver: `ConvDirectNaiveConvFwd`

## Workarounds tested

Spatial output tiling can make MIOpen select `GemmFwdRest`, but it is slower than the warmed full naive path:

| Original descriptor | Tiling | Tile descriptor | Solver | Time |
|---|---:|---|---|---:|
| `160-18-274-370 -> 160-16-272-368` | `2x1` | `160-18-138-370 -> 160-16-136-368` | `GemmFwdRest` | `80.835 sec` |
| `160-18-274-370 -> 160-16-272-368` | `2x2` | `160-18-138-186 -> 160-16-136-184` | `GemmFwdRest` | `40.251 sec` |
| `512-6-242-322 -> 256-4-240-320` | `2x1` | `512-6-122-322 -> 256-4-120-320` | `GemmFwdRest` | `90.176 sec` |
| `512-6-242-322 -> 256-4-240-320` | `2x2` | `512-6-122-162 -> 256-4-120-160` | `GemmFwdRest` | `41.895 sec` |

FP16 full descriptor was also checked for descriptor 1. It still used workspace `0` and `ConvDirectNaiveConvFwd`.

## Request

Please add or enable an optimized forward solver for these large BF16 NCDHW 3D convolution descriptors on gfx1151, or document why the current descriptor constraints force `ConvDirectNaiveConvFwd`.

The target is to avoid `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort` for the full descriptors without requiring spatial tiling that multiplies total work.
