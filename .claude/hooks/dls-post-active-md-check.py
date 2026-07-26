#!/usr/bin/env python3
"""PostToolUse hook (Write|Edit|MultiEdit): active.md 変更後の自動バリデーション"""

import json
import re
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    ti = data.get("tool_input", {})
    file_path = ti.get("file_path", ti.get("path", ""))

    # active.md の変更のみ対象
    if not file_path or not file_path.endswith(".dls/active.md"):
        return 0

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return 0

    issues = []
    warnings = []

    # エントリ数チェック
    entries = re.findall(r"^## DLS-\d+", content, re.MULTILINE)
    count = len(entries)

    if count > 30:
        warnings.append(
            f"📌 [エントリ数 {count}/目安 30] 軟上限超過（DLS-144）。"
            "Phase 集約期は許容、Phase 終了 or コミット集約イベントで "
            "05_archive_management.md Step 2.5 を実行してください。"
        )
    elif count >= 25:
        warnings.append(f"⚠️  [エントリ数警告] {count}/目安 30 に近づいています。")

    # sources フィールドの欠落チェック
    for block in re.split(r"\n(?=## DLS-)", content):
        m = re.match(r"## (DLS-\d+)", block.strip())
        if not m:
            continue
        entry_id = m.group(1)
        if not re.search(r"\*\*sources\*\*\s*:", block):
            issues.append(f"🚨 [sources 欠落] {entry_id}")
        elif re.search(r"\*\*sources\*\*\s*:\s*$", block, re.MULTILINE):
            issues.append(f"🚨 [sources 空] {entry_id}")

    # supersedes 済みエントリのアーカイブ漏れ
    for block in re.split(r"\n(?=## DLS-)", content):
        m_sup = re.search(r"\*\*supersedes\*\*\s*:\s*(DLS-\d+)", block)
        if m_sup:
            old_id = m_sup.group(1)
            if re.search(rf"^## {old_id}", content, re.MULTILINE):
                m_new = re.match(r"## (DLS-\d+)", block.strip())
                new_id = m_new.group(1) if m_new else "?"
                warnings.append(f"⚠️  [アーカイブ未完了] {old_id} (superseded by {new_id})")

    if issues or warnings:
        output = f"📋 DLS active.md チェック (エントリ数: {count}/目安 30)"
        if issues:
            output += "\n\n【要対応】\n" + "\n".join(issues)
        if warnings:
            output += "\n\n【確認推奨】\n" + "\n".join(warnings)
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
