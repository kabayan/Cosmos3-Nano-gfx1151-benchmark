#!/usr/bin/env bash
# DLS-126/127/128/140: Claude Code status line スクリプト
# 入力欄下に「モデル名 | ctx 使用率 [main/side label]」を 1 行表示する。
# stdin で Claude Code が JSON を渡してくる（statusLine 仕様）。
# settings.local.json で `statusLine.command` から呼ばれる。
#
# DLS-140: `.claude/.dls/sessions/current.md` が存在するプロジェクトでは、
# stdin JSON の `session_id` を current.md と照合し、以下のラベルを suffix 付与:
#   - main 一致: [main]（緑）
#   - active な side 一致: [side: <reason 60字>]（黄）
#   - merged/abandoned な side 一致: ラベルなし
#   - 不在で session_name あり: [?: <name 60字>]（薄灰）
# current.md がないプロジェクトでは現状の base 出力（model | ctx）のみで完全互換。
set -euo pipefail
input=$(cat)

# stdin JSON 抽出（jq 優先、なければ python3 fallback）
if command -v jq >/dev/null 2>&1; then
  model=$(printf '%s'      "$input" | jq -r '.model.display_name             // "Claude"')
  ctx_pct=$(printf '%s'    "$input" | jq -r '.context_window.used_percentage // ""')
  session_id=$(printf '%s' "$input" | jq -r '.session_id                     // ""')
  session_name=$(printf '%s' "$input" | jq -r '.session_name                 // ""')
else
  IFS='|' read -r model ctx_pct session_id session_name < <(printf '%s' "$input" | python3 -c '
import json, sys
d = json.load(sys.stdin)
m  = (d.get("model")          or {}).get("display_name", "Claude")
p  = (d.get("context_window") or {}).get("used_percentage", "")
si = d.get("session_id")   or ""
sn = d.get("session_name") or ""
print(m, p, si, sn, sep="|")
')
fi

# ctx 使用率（%）: used_percentage を直接使う。欠落時のみ「—」
if [ -n "${ctx_pct:-}" ] && [ "$ctx_pct" != "null" ]; then
  ctx_str=$(awk -v p="$ctx_pct" 'BEGIN{ printf "%d%%", p }')
else
  ctx_str="—"
fi

base_out=$(printf '\033[36m%s\033[0m | ctx \033[1m%s\033[0m' "$model" "$ctx_str")

# DLS-140: main/side label suffix（current.md があるプロジェクトのみ）
label=""
project_dir="${DLS_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-${GEMINI_PROJECT_DIR:-$PWD}}}"
current_md="$project_dir/.claude/.dls/sessions/current.md"
if [ -n "$session_id" ] && [ -f "$current_md" ]; then
  # main 一致 check（awk `\b` 非ポータブルのため `^## main[[:space:]]*$` を使う）
  main_id=$(awk '
    /^## main[[:space:]]*$/  { flag=1; next }
    /^## /                   { flag=0 }
    flag && /^- session_id:/ { print $3; exit }
  ' "$current_md")

  if [ -n "$main_id" ] && [ "$main_id" = "$session_id" ]; then
    label=$'\033[32m[main]\033[0m'
  else
    # side block 走査（最初に session_id 一致するブロックの status/reason を抽出）
    side_info=$(awk -v sid="$session_id" '
      in_block && /^## / { exit }
      /^## side: / && $3 == sid {
        in_block = 1
        matched  = 1
        next
      }
      /^## / { in_block = 0; next }
      in_block && /^- status:/ { status = $3 }
      in_block && /^- reason:/ { sub(/^- reason: */, ""); reason = $0 }
      END { if (matched) printf "%s|%s", status, reason }
    ' "$current_md")

    if [ -n "$side_info" ]; then
      # matched: status/reason 抽出 → active のみ label 付与
      side_status="${side_info%%|*}"
      side_reason="${side_info#*|}"
      if [ "$side_status" = "active" ] && [ -n "$side_reason" ]; then
        # 60 字 truncate
        side_reason_short=$(printf '%s' "$side_reason" | awk '{
          if (length($0) > 60) print substr($0, 1, 60) "…"
          else                  print $0
        }')
        label=$'\033[33m[side: '"$side_reason_short"$']\033[0m'
      fi
      # status != active なら matched でもラベルなし（DLS-018 と整合）
    elif [ -n "$session_name" ]; then
      # current.md にも未登録、session_name あり: unregistered fallback
      name_short=$(printf '%s' "$session_name" | awk '{
        if (length($0) > 60) print substr($0, 1, 60) "…"
        else                  print $0
      }')
      label=$'\033[90m[?: '"$name_short"$']\033[0m'
    fi
  fi
fi

# 出力組み立て（DLS-036: base 先、label suffix）
if [ -n "$label" ]; then
  printf '%s %s\n' "$base_out" "$label"
else
  printf '%s\n' "$base_out"
fi
