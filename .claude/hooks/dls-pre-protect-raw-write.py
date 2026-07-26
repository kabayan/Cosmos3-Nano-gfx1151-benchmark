#!/usr/bin/env python3
"""PreToolUse hook (Write|Edit|MultiEdit): .claude/.dls/raw/ 内の既存ファイルへの上書きをブロック"""

import json
import os
import re
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    ti = data.get("tool_input", {})
    file_path = ti.get("file_path", ti.get("path", ""))
    if not file_path:
        return 0

    # raw/ 以下のファイルかチェック
    if not re.search(r"(^|/)\.?(claude/)?\.dls/raw/.+", file_path):
        return 0

    tool_name = data.get("tool_name", "")

    # Write: 新規ファイルなら許可、既存ファイルへの上書きはブロック
    if tool_name == "Write":
        if os.path.isfile(file_path):
            print(
                f"🚫 DLS原則違反: .claude/.dls/raw/ 内の既存ファイルへの上書きは禁止です。\n"
                f"ファイル: {file_path}\n"
                "新しい内容は新規ファイルとして保存してください",
                file=sys.stderr,
            )
            return 2
        return 0

    # Edit / MultiEdit: 常にブロック
    print(
        f"🚫 DLS原則違反: .claude/.dls/raw/ 内のファイルの編集は禁止です。\n"
        f"ファイル: {file_path}\n"
        "  - 内容を追記したい場合: Bash の >> リダイレクトを使用してください\n"
        "  - 修正が必要な場合: 新規ファイルとして保存し、古いファイルは残してください",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
