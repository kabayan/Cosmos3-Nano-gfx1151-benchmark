#!/usr/bin/env python3
"""PreToolUse hook (Bash): git commit 時に DLS 整合性を強制する

チェック項目:
1. コミットメッセージに DLS-XXX 参照があること
2. .py ファイルをコミットする場合、active.md もステージされていること
"""

import json
import re
import subprocess
import sys


def _get_staged_files() -> list[str]:
    """ステージ済みファイル一覧を取得"""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().splitlines() if result.returncode == 0 else []
    except Exception:
        return []


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    command = data.get("tool_input", {}).get("command", "")
    if not command:
        return 0

    # git commit 以外は無視
    if not re.search(r"^\s*git\s+commit\b", command):
        return 0

    # amend はスキップ
    if re.search(r"\s--amend\b", command):
        return 0

    # チェック1: DLS-XXX パターンを検索
    if not re.search(r"DLS-\d+", command):
        print(
            "🚫 コミットメッセージに DLS エントリ参照がありません。\n"
            "メッセージに DLS-XXX を含めてください。\n\n"
            '例: git commit -m "DLS-002: セッション有効期限の制限を実装"\n\n'
            "理由:\n"
            '  - git log --grep="DLS-XXX" による判断→実装の逆引きを可能にするため\n'
            "  - .claude/.dls/active.md で現在のエントリ番号を確認してください",
            file=sys.stderr,
        )
        return 2

    # チェック2: .py をコミットするなら active.md もステージされているか
    staged = _get_staged_files()
    has_py = any(f.endswith(".py") for f in staged)
    has_active_md = any("active.md" in f for f in staged)

    if has_py and not has_active_md:
        print(
            "🚫 .py ファイルがステージされていますが active.md が含まれていません。\n"
            "実装判断を伴うコミットでは、先に DLS エントリを作成・更新してください。\n\n"
            "判断が不要な軽微な修正の場合:\n"
            "  active.md を変更なしでステージ（git add .claude/.dls/active.md）するか、\n"
            "  このチェックの意図を確認してください。",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
