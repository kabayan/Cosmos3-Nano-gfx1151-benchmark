#!/usr/bin/env python3
"""PreToolUse hook (Write|Edit|MultiEdit):

編集対象ファイルに関連する DLS エントリを動的にコンテキストへ注入する。
where フィールドとファイルパスをベストエフォートで照合し、関連エントリのみを提示する。

設計意図 (SOURCE-005 対策):
- CLAUDE.md 常時注入の DLS 運用ルールを削減できるように、編集直前に必要な判断情報を再注入する
- 位置バイアスを回避（アクション直前に注入されるため、希釈されにくい）
- 無関係な編集ではコンテキストを消費しない
"""

import json
import os
import re
import sys


def parse_entries(content: str) -> list[dict]:
    """active.md から DLS エントリをパースする"""
    entries: list[dict] = []
    # ## DLS-XXX で始まるブロックに分割
    blocks = re.split(r"\n(?=## DLS-)", content)
    for block in blocks:
        m = re.match(r"^## (DLS-\S+)", block)
        if not m:
            continue
        entry_id = m.group(1)
        what_m = re.search(r"-\s*\*\*what\*\*:\s*(.+?)(?=\n-\s*\*\*|\Z)", block, re.DOTALL)
        where_m = re.search(r"-\s*\*\*where\*\*:\s*(.+?)(?=\n-\s*\*\*|\Z)", block, re.DOTALL)
        has_low = "confidence: low" in block
        entries.append(
            {
                "id": entry_id,
                "what": (what_m.group(1).strip() if what_m else "")[:200],
                "where": (where_m.group(1).strip() if where_m else "")[:300],
                "has_low_confidence": has_low,
            }
        )
    return entries


def file_matches_where(file_path: str, where: str, project_dir: str) -> bool:
    """file_path が where に関連するか判定（ベストエフォート照合）"""
    if not file_path or not where:
        return False
    # ファイル名（basename）が where に含まれる
    basename = os.path.basename(file_path)
    if basename and basename in where:
        return True
    # 絶対パスが where に含まれる
    if file_path in where:
        return True
    # プロジェクトルート相対パス・親ディレクトリで照合
    if project_dir and file_path.startswith(project_dir):
        rel_path = os.path.relpath(file_path, project_dir)
        if rel_path in where:
            return True
        rel_dir = os.path.dirname(rel_path)
        if rel_dir and rel_dir in where:
            return True
    return False


def emit_context(context: str) -> None:
    """hookSpecificOutput JSON を stdout に出力してコンテキスト注入する"""
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "additionalContext": context,
        }
    }
    print(json.dumps(output, ensure_ascii=False))


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return 0

    # DLS-129/139: CLI 抽象化のため DLS_PROJECT_DIR を最優先で参照する
    project_dir = (
        os.environ.get("DLS_PROJECT_DIR")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.environ.get("GEMINI_PROJECT_DIR")
        or ""
    )
    if not project_dir:
        return 0

    active_md = os.path.join(project_dir, ".claude", ".dls", "active.md")
    if not os.path.isfile(active_md):
        return 0

    with open(active_md, encoding="utf-8") as f:
        content = f.read()

    entries = parse_entries(content)
    matches = [e for e in entries if file_matches_where(file_path, e["where"], project_dir)]
    if not matches:
        return 0

    # 注入するコンテキストを構築
    try:
        rel_path = os.path.relpath(file_path, project_dir)
    except ValueError:
        rel_path = file_path
    lines = [f"📋 関連DLSエントリ ({rel_path}):"]
    for e in matches:
        flag = "  ⚠️ confidence: low を含む" if e["has_low_confidence"] else ""
        lines.append(f"- {e['id']}: {e['what']}{flag}")
        lines.append(f"  where: {e['where']}")
    lines.append("")
    lines.append("これらの判断・棄却案を踏まえて実装し、新たな判断が発生したらコミット前にDLSエントリを追加すること。")

    emit_context("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
