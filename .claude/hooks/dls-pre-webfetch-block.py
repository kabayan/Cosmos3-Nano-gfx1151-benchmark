#!/usr/bin/env python3
"""PreToolUse hook (WebFetch): WebFetch を playwright-cli に誘導してブロックする。

DLS-124/157: playwright-cli が PATH に存在する場合のみ exit 2 でブロックする。
- WebFetch（URL 本文取得）→ playwright-cli に誘導（DLS-124 主軸、token-efficient）
  - 主: `playwright-cli open <URL>` → `playwright-cli snapshot`（公式 @playwright/cli）
  - 副: Playwright MCP / Chrome MCP（重量・プロファイル衝突あり、CLI 不在時のみ）
  - 根拠: gemini --prompt は paywall サイトで作話する実例あり（DLS-121 / feedback_paywall_url_fetch.md）

WebSearch は内蔵 API を使用（DLS-157: gemini-cli の google_web_search 廃止予定により
SOURCE-028 由来の gemini-cli 一律化方針を撤回）。本フックは WebFetch 専用に縮退。

playwright-cli 不在環境では exit 0 で素通り（DLS-120 思想踏襲 — 代替手段が無い環境ではブロックしない）。
"""

import shutil
import sys


def main() -> int:
    if not shutil.which("playwright-cli"):
        return 0  # playwright-cli 未導入環境では WebFetch を許容（DLS-120 思想踏襲）

    print(
        "🚫 WebFetch 禁止: ページ本文の取得は playwright-cli を使用してください（DLS-124 主軸）。\n"
        "  - 主: `playwright-cli open <URL>` → `playwright-cli snapshot`（公式 @playwright/cli、token-efficient）\n"
        "  - 副: Playwright MCP / Chrome MCP（重量、プロファイル衝突あり、CLI 不在時のみ）\n"
        "  - 根拠: gemini --prompt は paywall サイトで作話する実例あり（DLS-121 / feedback_paywall_url_fetch.md）。\n"
        "    公式 README 曰く CLI は MCP より低トークン（large tool schemas を context に載せない）\n"
        "  - セットアップ: `npm i -g @playwright/cli@latest`（docs/setup.md#playwright-cli）",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
