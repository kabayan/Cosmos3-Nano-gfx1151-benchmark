#!/usr/bin/env python3
"""SessionStart hook: session_id ドリフト検出 + 条件付き自動 re-claim (DLS-035 + DLS-147).

現 session_id が `.claude/.dls/sessions/current.md` の main_id とも全 active
side_id とも一致しない (= drift) 場合の挙動:

  - `SIDE_SESSION` env 未設定 (= main 経路) → `## main` ブロックを現 session_id に
    自動上書きし、auto-reclaim 通知を additionalContext で返す (DLS-147)。
    並走 main cc は後発が勝つ仕様（受容、各プロジェクトの自己診断 skill で確認）。
  - `SIDE_SESSION` env 設定済 (= side 経路) → 書き換えず警告のみ返す（DLS-035 既存挙動）。

current.md は ksd/navi 系プロジェクトのスキーマ (DLS-002 系)。スキーマがない他プロジェクトでは
ファイル不在で early return し副作用なし。
"""

import datetime
import json
import os
import re
import sys


def main():
    if sys.stdin.isatty():
        return 0
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    session_id = payload.get("session_id")
    if not session_id:
        return 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_md = os.path.join(script_dir, "..", ".dls", "sessions", "current.md")

    if not os.path.isfile(current_md):
        return 0

    try:
        with open(current_md, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return 0

    main_match = re.search(
        r"^## main\b.*?^- session_id:\s*(\S+)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    main_id = main_match.group(1) if main_match else None

    active_side_ids = []
    for m in re.finditer(
        r"^## side: ([0-9a-fA-F-]+)\b(.*?)(?=^## |\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    ):
        sid = m.group(1)
        block = m.group(0)
        status_m = re.search(r"^- status:\s*(\S+)", block, re.MULTILINE)
        if status_m and status_m.group(1) == "active":
            active_side_ids.append(sid)

    if session_id == main_id or session_id in active_side_ids:
        return 0

    side_env = os.environ.get("SIDE_SESSION", "")

    # DLS-147: SIDE_SESSION env が未設定 (= main 経路) なら自動 re-claim
    # 並走 main cc は後発が勝つ仕様（受容、各プロジェクトの自己診断 skill で確認）
    if not side_env:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        new_main_block = (
            "## main\n\n"
            f"- session_id: {session_id}\n"
            "- name: main\n"
            f"- started_at: {ts}\n"
            "- status: active\n"
            "- note: auto-reclaimed by SessionStart hook (DLS-147)\n"
        )
        if "## main" in content:
            new_content = re.sub(
                r"## main.*?(?=\n## |\Z)",
                new_main_block + "\n",
                content,
                count=1,
                flags=re.DOTALL,
            )
        else:
            new_content = content.rstrip() + "\n\n" + new_main_block + "\n"
        try:
            with open(current_md, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception:
            pass
        else:
            old_label = (main_id[:8] + "…") if main_id else "(未登録)"
            reclaim_msg = "\n".join([
                "✅ session_id ドリフト自動 re-claim 完了 (DLS-147)",
                f"  旧 main_id: {old_label}",
                f"  新 main_id: {session_id[:8]}…",
                "  current.md の ## main を本セッションに更新しました。"
                "statusline は次回 refresh で [main] 表示に切り替わります。",
                "  注意: 並走 main cc がある場合は後発（本タブ）が勝つ仕様です。"
                "各プロジェクトの自己診断 skill で他タブが影響を受けていないか確認してください。",
            ])
            print(json.dumps({"additionalContext": reclaim_msg}))
            return 0
        # 書き込み失敗時は既存の warning ロジックにフォールスルー

    hint_lines = []
    if side_env and side_env != session_id:
        hint_lines.append(
            f"  SIDE_SESSION env (`{side_env[:8]}…`) と現 session_id (`{session_id[:8]}…`) が乖離。"
        )
        hint_lines.append(
            "  → cc が --fork-session / --resume 時に --session-id を新採番した可能性 (DLS-035)。"
        )
    elif main_id and session_id != main_id:
        hint_lines.append(
            f"  main_id (`{main_id[:8]}…`) と現 session_id (`{session_id[:8]}…`) が乖離。"
        )
        hint_lines.append(
            "  → cc が --resume で session_id を新採番した可能性 (DLS-007/DLS-035)。"
        )
    elif not main_id:
        hint_lines.append("  current.md に main_id が未登録 (まだ claim skill 未実行)。")

    warning_lines = [
        "⚠️  session_id ドリフト検出 (DLS-035)",
        f"  現 session_id: {session_id}",
        "  current.md に未登録 (main / active side のいずれにも一致なし)。",
    ]
    warning_lines.extend(hint_lines)
    warning_lines.extend([
        "  対処:",
        "    - このタブを main として登録: /navi-claim-main (or 各プロジェクト同等 skill)",
        "    - statusLine の `[?: ...]` 表示は本ドリフトの想定挙動 (DLS-019)",
    ])
    warning = "\n".join(warning_lines)

    print(json.dumps({"additionalContext": warning}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
