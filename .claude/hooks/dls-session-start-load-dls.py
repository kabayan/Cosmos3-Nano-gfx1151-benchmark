#!/usr/bin/env python3
"""SessionStart hook: DLS active.md をセッション開始時にコンテキストへ注入する。

DLS-122: `.migration_state.json` に `update_pending` フラグがあるとき、
additionalContext の先頭に最優先で `/dls-update` 実行指示を注入し、
stderr にもユーザー向け警告を出力する（init.py 後の必須メンテナンス強制）。
"""

import json
import os
import re
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    active_md = os.path.join(script_dir, "..", ".dls", "active.md")
    state_path = os.path.join(script_dir, "..", ".dls", ".migration_state.json")

    # update_pending フラグの検出（DLS-122）
    update_pending = None
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                update_pending = json.load(f).get("update_pending")
        except Exception:
            update_pending = None

    pending_block = ""
    if update_pending:
        reason = update_pending.get("reason", "unknown")
        triggered_at = update_pending.get("triggered_at", "unknown")
        pending_block = (
            "🚨🚨🚨 DLS UPDATE REQUIRED 🚨🚨🚨\n"
            f"理由: {reason}\n"
            f"トリガー時刻: {triggered_at}\n"
            "**他のいかなる作業よりも先に `/dls-update` を実行してください。**\n"
            "ユーザーの最初のリクエストに着手する前に、必ずこのコマンドを実行する必要があります。\n"
            "完了時に `update_pending` フラグは自動で降ろされます。\n"
            "=======================================\n\n"
        )
        # ユーザー画面にも見える形で警告
        print(pending_block, file=sys.stderr)

    if not os.path.isfile(active_md):
        print(json.dumps({
            "additionalContext": pending_block + "⚠️ .claude/.dls/active.md が見つかりません。"
        }))
        return 0

    with open(active_md, encoding="utf-8") as f:
        content = f.read()

    entry_count = len(re.findall(r"^## DLS-", content, re.MULTILINE))
    low_confidence = content.count("confidence: low")

    warnings = ""
    if entry_count >= 25:
        warnings += f"\n⚠️ active.md のエントリ数が {entry_count}/目安 30 に達しています（DLS-144 軟上限）。"
    if low_confidence > 0:
        warnings += f"\n⚠️ confidence: low の未検証前提が {low_confidence} 件あります。"

    context = (
        f"{pending_block}"
        f"=== DLS CONTEXT (自動注入) ===\n"
        f"エントリ数: {entry_count}/目安 30\n\n"
        f"{content}\n"
        f"=== DLS CONTEXT END ==="
    )
    if warnings:
        context += f"\n{warnings}"

    print(json.dumps({"additionalContext": context}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
