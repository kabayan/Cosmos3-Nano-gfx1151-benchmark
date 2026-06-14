import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


DESCRIPTORS = {
    "policy_160_18_274_370_to_160": {
        "input": (1, 160, 18, 274, 370),
        "weight": (160, 160, 3, 3, 3),
    },
    "policy_512_6_242_322_to_256": {
        "input": (1, 512, 6, 242, 322),
        "weight": (256, 512, 3, 3, 3),
    },
}


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def output_partitions(total: int, parts: int) -> list[tuple[int, int]]:
    base = total // parts
    rem = total % parts
    ranges = []
    start = 0
    for idx in range(parts):
        size = base + (1 if idx < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def tile_slices(output_shape: tuple[int, int, int], tile_h: int, tile_w: int) -> list[tuple[slice, slice, slice]]:
    out_d, out_h, out_w = output_shape
    tiles = []
    for h0, h1 in output_partitions(out_h, tile_h):
        for w0, w1 in output_partitions(out_w, tile_w):
            tiles.append((slice(0, out_d), slice(h0, h1), slice(w0, w1)))
    return tiles


def input_crop_for_output_tile(tile: tuple[slice, slice, slice], kernel: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    d, h, w = tile
    kd, kh, kw = kernel
    return (
        slice(d.start, d.stop + kd - 1),
        slice(h.start, h.stop + kh - 1),
        slice(w.start, w.stop + kw - 1),
    )


def run_full_conv(x: torch.Tensor, weight: torch.Tensor, repeats: int) -> dict:
    times = []
    out = None
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        out = F.conv3d(x, weight)
        synchronize()
        times.append(time.perf_counter() - start)
    return {
        "times_sec": times,
        "avg_sec": sum(times) / len(times),
        "output_shape": tuple(out.shape) if out is not None else None,
    }


def run_tiled_conv(x: torch.Tensor, weight: torch.Tensor, repeats: int, tile_h: int, tile_w: int) -> dict:
    _, _, in_d, in_h, in_w = x.shape
    out_channels, _, kd, kh, kw = weight.shape
    out_shape = (in_d - kd + 1, in_h - kh + 1, in_w - kw + 1)
    tiles = tile_slices(out_shape, tile_h, tile_w)
    times = []
    tile_shapes = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        outputs = []
        for tile in tiles:
            d, h, w = input_crop_for_output_tile(tile, (kd, kh, kw))
            x_tile = x[:, :, d, h, w]
            y_tile = F.conv3d(x_tile, weight)
            outputs.append(y_tile)
            tile_shapes.append(tuple(x_tile.shape))
        synchronize()
        times.append(time.perf_counter() - start)
        del outputs
    return {
        "times_sec": times,
        "avg_sec": sum(times) / len(times),
        "output_shape": (x.shape[0], out_channels, *out_shape),
        "tile_count": len(tiles),
        "unique_input_tile_shapes": sorted({shape for shape in tile_shapes}),
    }


def run_case(name: str, descriptor: dict, dtype_name: str, repeats: int, tile_h: int, tile_w: int) -> dict:
    dtype = DTYPES[dtype_name]
    device = torch.device("cuda")
    input_shape = descriptor["input"]
    weight_shape = descriptor["weight"]
    torch.manual_seed(0)
    x = torch.randn(input_shape, device=device, dtype=dtype)
    weight = torch.randn(weight_shape, device=device, dtype=dtype)
    synchronize()

    if tile_h == 1 and tile_w == 1:
        result = run_full_conv(x, weight, repeats)
        mode = "full"
    else:
        result = run_tiled_conv(x, weight, repeats, tile_h, tile_w)
        mode = f"tile_h{tile_h}_w{tile_w}"

    result.update(
        {
            "descriptor": name,
            "dtype": dtype_name,
            "mode": mode,
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "torch_version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "device": torch.cuda.get_device_name(0),
            "max_memory_allocated": torch.cuda.max_memory_allocated(),
            "max_memory_reserved": torch.cuda.max_memory_reserved(),
        }
    )
    del x, weight
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--descriptor", choices=sorted(DESCRIPTORS), action="append", default=[])
    parser.add_argument("--dtype", choices=sorted(DTYPES), action="append", default=[])
    parser.add_argument("--tile", action="append", default=[], help="Output tiling as HxW. 1x1 means full conv.")
    parser.add_argument("--skip-full", action="store_true", help="Skip 1x1/full conv entries.")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptors = args.descriptor or list(DESCRIPTORS)
    dtypes = args.dtype or ["bf16"]
    tiles = args.tile or ["1x1"]
    if args.skip_full:
        tiles = [tile for tile in tiles if tile.lower() != "1x1"]

    results = []
    for descriptor_name in descriptors:
        for dtype_name in dtypes:
            for tile in tiles:
                tile_h, tile_w = [int(part) for part in tile.lower().split("x", 1)]
                result = run_case(
                    descriptor_name,
                    DESCRIPTORS[descriptor_name],
                    dtype_name,
                    args.repeats,
                    tile_h,
                    tile_w,
                )
                print(json.dumps(result, indent=2), flush=True)
                results.append(result)

    output = {"results": results}
    (out_dir / "large_conv3d_probe.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
