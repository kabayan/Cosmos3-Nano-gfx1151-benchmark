#!/usr/bin/env python3
"""SessionEnd hook (matcher: clear): /clear 直前の議論ノート未保存最終確認 (DLS-146)

二段防衛の「最終ゲート」を担う。Stop hook の dls-stop-reminder.py が CC 応答完了時の
予防警告を担当し、本 hook は /clear 実行時の最終確認警告を出す。

- reason フィールドが "clear" 以外（logout / prompt_input_exit / other）はスキップ
- 当日 DLS エントリ追加あり AND 当日 chat/discussion ファイル未作成 → stderr 警告
- decision: block は SessionEnd では非対応のため、stderr 警告のみ（ユーザー判断に委ねる）
"""

import json
import os
import re
import sys
from datetime import datetime


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    if data.get("reason") != "clear":
        return 0

    project_dir = (
        os.environ.get("DLS_PROJECT_DIR")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.environ.get("GEMINI_PROJECT_DIR")
        or os.getcwd()
    )
    active_md = os.path.join(project_dir, ".claude/.dls/active.md")
    raw_dir = os.path.join(project_dir, ".claude/.dls/raw")
    today = datetime.now().strftime("%Y-%m-%d")
    today_compact = datetime.now().strftime("%Y%m%d")

    if not os.path.exists(active_md):
        return 0

    with open(active_md, encoding="utf-8") as f:
        content = f.read()
    today_entries = len(re.findall(rf"date.*{today}", content))

    today_notes = 0
    if os.path.isdir(raw_dir):
        for fname in os.listdir(raw_dir):
            if (
                fname.startswith(f"{today_compact}_chat_")
                or fname.startswith(f"{today_compact}_discussion_")
            ) and fname.endswith(".md"):
                today_notes += 1

    if today_entries > 0 and today_notes == 0:
        print(
            f"\n⚠️  /clear が実行されようとしています。議論ノートが未作成です "
            f"(DLS エントリ {today_entries} 件追加 / 議論ノート 0 件)。\n"
            f"   この時点で /clear すると議論は復活できません。\n"
            f"   中断して /dls-commit を実行することを強く推奨します（DLS-146）。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
