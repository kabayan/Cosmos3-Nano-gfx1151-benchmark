#!/usr/bin/env python3
"""PreToolUse hook (Bash): .claude/.dls/raw/ への破壊的操作をブロックする"""

import json
import re
import sys

RAW_PATTERN = r"(\.claude/)?\.dls/raw/"


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    command = data.get("tool_input", {}).get("command", "")
    if not command:
        return 0

    # rm で raw/ 以下を削除
    if re.search(rf"rm\s.*{RAW_PATTERN}", command):
        return block("rm による .claude/.dls/raw/ への削除操作")

    # mv で raw/ 以下のファイルを外へ移動
    if re.search(rf"mv\s.*{RAW_PATTERN}", command):
        # raw/内同士の移動はOK
        parts = command.split()
        raw_count = sum(1 for p in parts if re.search(RAW_PATTERN, p))
        if raw_count < 2:
            return block("mv による .claude/.dls/raw/ からのファイル移動")

    # truncate で raw/ 以下を空に
    if re.search(rf"truncate.*{RAW_PATTERN}", command):
        return block("truncate による .claude/.dls/raw/ の内容削除")

    # > リダイレクトで既存ファイルを上書き（>> はOK）
    if re.search(r"(?<!>)>\s*(?:\.claude/)?\.dls/raw/\S+\.\w+", command):
        return block("> リダイレクトによる .claude/.dls/raw/ 既存ファイルの上書き")

    return 0


def block(reason: str) -> int:
    print(
        f"🚫 DLS原則違反: {reason}\n"
        ".claude/.dls/raw/ は原本ストアです。追記のみ許可されています。\n"
        "  - 新しい内容は新規ファイルとして保存してください\n"
        "  - 既存ファイルの削除・上書きは禁止です",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
