import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a_payload = torch.load(args.a, map_location="cpu")
    b_payload = torch.load(args.b, map_location="cpu")
    a = a_payload["output"].float()
    b = b_payload["output"].float()
    diff = (a - b).abs()
    result = {
        "a": args.a,
        "b": args.b,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "a_dtype": str(a_payload["output"].dtype),
        "b_dtype": str(b_payload["output"].dtype),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "nonzero": int((diff != 0).sum().item()),
        "count": int(diff.numel()),
        "a_cold_reference_baseline_seconds": a_payload.get("cold_reference_baseline_seconds"),
        "b_cold_reference_baseline_seconds": b_payload.get("cold_reference_baseline_seconds"),
        "a_warm_baseline_seconds": a_payload.get("warm_baseline_seconds"),
        "b_warm_baseline_seconds": b_payload.get("warm_baseline_seconds"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
