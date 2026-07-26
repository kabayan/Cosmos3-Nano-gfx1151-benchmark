#!/usr/bin/env python3
"""PreToolUse hook (Write|Edit|MultiEdit): E2 = INV-Q の機械強制（PoC / live 非投入）

dls-exec の不変条件 INV-Q（再定義版）を hook で強制する:
  「定量 verdict（computed_from: マーカー付き）を記録する write は、
   当該応答ターンで実際に code 実行ツール（Bash）が invoke されていなければブロックする」

E3 prose ゲートは cheap モデルに binding でない（P1 検証で haiku が computed_from を捏造）。
prose は「artifact が埋まっているか」しか検査できず「本物の実行出力か」を検査不能。
本 hook は transcript を out-of-band で走査し「当該ターンに Bash tool_use があったか」を検証する。

設計方針:
- positive 検出: computed_from: マーカーがある write のみ対象。無ければ素通し。
- 誤爆防止: file_path が .claude/.dls/raw/ or docs/ 配下なら対象外（設計 doc / raw ログ内の引用）。
- 粗い版: 現ターンの Bash tool_use 数 > 0 で pass（頑健版 = stdout 照合は strengthen 保留）。
- fail-open: stdin parse / transcript 特定 / ターン起点不明 は return 0（best-effort enforce）。
"""

import glob
import json
import os
import re
import sys


def _extract_write_content(tool_input: dict) -> str:
    """Write/Edit/MultiEdit の書き込み内容を 1 つの文字列に集約する"""
    parts = []
    # Write
    if "content" in tool_input:
        parts.append(str(tool_input.get("content", "")))
    # Edit
    if "new_string" in tool_input:
        parts.append(str(tool_input.get("new_string", "")))
    # MultiEdit
    for edit in tool_input.get("edits", []) or []:
        parts.append(str(edit.get("new_string", "")))
    return "\n".join(parts)


def _is_excluded_path(file_path: str) -> bool:
    """設計 doc / raw ログは verdict 記録ではない → computed_from: 引用を誤爆させない"""
    if not file_path:
        return False
    norm = file_path.replace("\\", "/")
    return ("/.dls/raw/" in norm or norm.endswith("raw")
            or "/docs/" in norm or "/docs/" in ("/" + norm))


def _resolve_transcript(data: dict) -> str | None:
    """transcript_path（あれば最優先）→ 無ければ session_id から glob"""
    tp = data.get("transcript_path")
    if tp and os.path.isfile(tp):
        return tp

    session_id = data.get("session_id")
    cwd = data.get("cwd") or os.getcwd()
    if not session_id:
        return None

    encoded = cwd.replace("/", "-")
    home = os.path.expanduser("~")
    base = os.path.join(home, ".claude", "projects", encoded)

    # session_id 直当てを最優先
    candidates = glob.glob(os.path.join(base, f"{session_id}.jsonl"))
    candidates += glob.glob(os.path.join(base, "**", f"{session_id}.jsonl"), recursive=True)
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _bash_uses_in_current_turn(transcript_path: str) -> int | None:
    """transcript を走査し、現ターン（末尾の人間プロンプト以降）の Bash tool_use 数を返す。
    ターン起点が特定できなければ None（fail-open 用）。"""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
    except Exception:
        return None

    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    # 末尾から遡り、message.content が str の user 行 = 人間プロンプト = ターン起点
    start_idx = None
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        if r.get("type") != "user":
            continue
        content = r.get("message", {}).get("content")
        if isinstance(content, str):  # list の user 行は tool_result なので除外
            start_idx = i
            break
    if start_idx is None:
        return None

    # ターン起点以降の assistant の Bash tool_use をカウント（attachment 行は type で自然に弾かれる）
    count = 0
    for r in rows[start_idx + 1:]:
        if r.get("type") != "assistant":
            continue
        for item in r.get("message", {}).get("content", []) or []:
            if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("name") == "Bash":
                count += 1
    return count


BLOCK_MESSAGE = """🚫 INV-Q 違反: 定量 verdict（computed_from: 付き）を記録しようとしていますが、
   現在の応答ターンで実際の code 実行ツール（Bash）の invoke が観測されません（tool_uses=0）。
   computed_from の数値は実行されていない可能性があります（暗算 / 捏造）。

   → dls-exec STEP 4（COMPUTE）に戻り、判断の数値を産む code を Bash で実行し、
     その stdout を computed_from に貼ってから verdict を記録してください。

   （この検査は transcript の現ターン tool_use を out-of-band で検証しています。
     prose ゲートでは捏造を検出できないため hook で強制しています。）"""


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # fail-open

    tool_input = data.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")

    # 誤爆防止: 設計 doc / raw ログは対象外
    if _is_excluded_path(file_path):
        return 0

    content = _extract_write_content(tool_input)
    if not content:
        return 0

    # positive 検出: computed_from: マーカーが無ければ定量 verdict でない → 素通し
    if not re.search(r"computed_from\s*:", content):
        return 0

    transcript_path = _resolve_transcript(data)
    if not transcript_path:
        return 0  # fail-open: transcript 特定不可

    bash_count = _bash_uses_in_current_turn(transcript_path)
    if bash_count is None:
        return 0  # fail-open: ターン起点不明

    if bash_count == 0:
        print(BLOCK_MESSAGE, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
