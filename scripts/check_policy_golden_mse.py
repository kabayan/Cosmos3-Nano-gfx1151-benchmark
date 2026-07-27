"""Policy 出力を公式 golden action と照合する（DLS-011 / DLS-012）。

メトリックは cosmos_framework.inference.metrics.compute_action_mse と同一式
（全要素の mean((gt - pred)^2)）。合格ラインは inputs/omni/action_policy_robot.json の
extra.golden_mse_max = 0.05（記事の報告値は 0.013194）。

使い方:
    python scripts/check_policy_golden_mse.py result/mainline_full_v4_20260726 [...]
    python scripts/check_policy_golden_mse.py            # result/*/action_policy_robot を全走査

依存: 標準ライブラリのみ（ホスト実行可）。golden は初回にダウンロードして
result/golden_action_bridge_20260501_0.json にキャッシュする。
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    "2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/bridge_20260501_0.json"
)
GOLDEN_CACHE = REPO / "result" / "golden_action_bridge_20260501_0.json"
GOLDEN_MSE_MAX = 0.05


def load_golden() -> list[list[float]]:
    if not GOLDEN_CACHE.exists():
        GOLDEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GOLDEN_URL) as r:
            GOLDEN_CACHE.write_bytes(r.read())
    return json.loads(GOLDEN_CACHE.read_text())


def action_of(run_dir: Path) -> list[list[float]] | None:
    p = run_dir / "action_policy_robot" / "sample_outputs.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["outputs"][0]["content"]["action"]
    except (KeyError, IndexError, TypeError):
        return None


def mse(a: list[list[float]], b: list[list[float]]) -> float:
    n = 0
    s = 0.0
    for ra, rb in zip(a, b, strict=True):
        for x, y in zip(ra, rb, strict=True):
            s += (x - y) ** 2
            n += 1
    return s / n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run ディレクトリ（省略時は result/* を走査）")
    args = parser.parse_args()

    golden = load_golden()
    run_dirs = [Path(r) for r in args.runs] or sorted(REPO.glob("result/*"))

    any_fail = False
    for run_dir in run_dirs:
        act = action_of(run_dir)
        if act is None:
            continue
        m = mse(golden, act)
        verdict = "PASS" if m < GOLDEN_MSE_MAX else "FAIL"
        any_fail |= verdict == "FAIL"
        print(f"{run_dir.name:40s} golden MSE = {m:.6f}  {verdict}")
    print(f"(threshold golden_mse_max = {GOLDEN_MSE_MAX} / article reported 0.013194)")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
