#!/usr/bin/env python3
"""SessionStart hook (clear): /clear 後に tasks/next.md をコンテキストへ注入する"""

import json
import os
import sys
import time


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    next_md = os.path.join(project_dir, "tasks", "next.md")

    if not os.path.isfile(next_md):
        return 0

    with open(next_md, encoding="utf-8") as f:
        content = f.read()

    # ファイルの経過時間を確認
    mtime = os.path.getmtime(next_md)
    age_hours = int((time.time() - mtime) / 3600)

    warnings = ""
    if age_hours >= 48:
        warnings = f"\n⚠️ tasks/next.md は {age_hours} 時間前に作成されました。内容が古い可能性があります。"

    context = (
        "=== 前セッション引き継ぎ (自動注入) ===\n"
        "作業を再開する場合は /continue を実行してください。\n\n"
        f"{content}\n"
        "=== 引き継ぎ情報 END ==="
    )
    if warnings:
        context += warnings

    print(json.dumps({"additionalContext": context}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
