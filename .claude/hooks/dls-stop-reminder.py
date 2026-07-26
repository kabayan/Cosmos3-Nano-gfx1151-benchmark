#!/usr/bin/env python3
"""Stop hook: CC 応答完了時の **予防警告**（DLS-146 二段防衛の前段）

CC が応答を完了するたびに発火する。当日の DLS 記録漏れ・議論ノート未作成・confidence: low
前提を検知して stderr に予防警告を出力する。/clear 時の最終確認ゲートは
`dls-session-end-reminder.py`（SessionEnd hook, matcher: clear）が担う。
"""

import json
import os
import re
import sys
from datetime import date


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    reason = data.get("reason", "unknown")
    if reason in ("error", "sigint"):
        return 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    active_md = os.path.join(script_dir, "..", ".dls", "active.md")
    raw_dir = os.path.join(script_dir, "..", ".dls", "raw")

    if not os.path.isfile(active_md):
        return 0

    with open(active_md, encoding="utf-8") as f:
        content = f.read()

    today = date.today().isoformat()
    today_compact = today.replace("-", "")
    entry_count = len(re.findall(r"^## DLS-", content, re.MULTILINE))
    today_entries = len(re.findall(rf"date.*{today}", content))
    low_confidence = content.count("confidence: low")

    today_chat_notes = 0
    if os.path.isdir(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.startswith(f"{today_compact}_chat_") and fname.endswith(".md"):
                today_chat_notes += 1

    reminders = []

    if today_entries == 0:
        reminders.append(
            f"📝 今日（{today}）の作業で DLS エントリが記録されていません。\n"
            "   設計判断があれば記録してください。"
        )

    if today_entries > 0 and today_chat_notes == 0:
        reminders.append(
            f"⚠️  今日 DLS エントリが {today_entries} 件追加されましたが、"
            f"議論ノート（`.claude/.dls/raw/{today_compact}_chat_*.md`）が未作成です。\n"
            "   設計議論・中立的再評価・外部事例参照があった場合は、`/dls-commit` の Phase 1.6 で議論ノートを作成してください。\n"
            "   ⚠️ 議論ノートが消失すると **復活できません**。`/clear` / cc 再起動の前に必ず確認してください。"
        )

    if low_confidence > 0:
        reminders.append(f"⚠️  confidence: low の未検証前提が {low_confidence} 件あります。")

    if entry_count >= 25:
        reminders.append(f"⚠️  active.md のエントリ数が {entry_count}/目安 30 です（DLS-144 軟上限、Phase 終了 or コミット集約イベントで整理検討）。")

    if reminders:
        print("── DLS セッション終了チェック ──")
        print("\n".join(reminders))
        print("──────────────────────────────")

    return 0


if __name__ == "__main__":
    sys.exit(main())
