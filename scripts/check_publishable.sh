#!/usr/bin/env bash
# Leak scan: makes sure nothing private is about to be committed to this PUBLIC
# repository. Scans every tracked + untracked-but-not-ignored file.
#
#   scripts/check_publishable.sh          # scan; non-zero exit = leak
#   scripts/check_publishable.sh --list   # list the files that would be scanned
#   scripts/install_hooks.sh              # run this automatically before each commit
#
# Built-in patterns catch the generic tells (private IPs, home directories,
# tool-generated trailers, private docs). Site-specific tells — your ssh
# aliases, host names, project names, local tool files — go in
# local/private_patterns.txt (gitignored; one extended regex per line,
# # comments allowed).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Files that are scanned but are allowed to mention the patterns (this script
# and the example env).
ALLOW='^(scripts/check_publishable\.sh|scripts/install_hooks\.sh|scripts/local\.env\.example)$'

PAT_IP='\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b'
PAT_PATH='/home/[a-z][a-z0-9_-]*/|/Users/[a-z][a-z0-9_-]*/'
PAT_DOC='docs/private/[A-Za-z0-9_./-]+\.md'
# Tool-generated attribution: bot trailers, session links, "generated with" banners.
PAT_TRAIL='Co-Authored-By:.*(noreply@|\[bot\])|[A-Za-z]+-Session: *https?://|[Gg]enerated with \[?[A-Z][a-z]+ [A-Z]'
PAT="${PAT_IP}|${PAT_PATH}|${PAT_DOC}|${PAT_TRAIL}"

PRIV="${RCDL_PRIVATE_PATTERNS:-local/private_patterns.txt}"
if [ -f "$PRIV" ]; then
  extra="$(grep -vE '^\s*(#|$)' "$PRIV" | paste -sd'|' -)"
  [ -n "$extra" ] && PAT="${PAT}|${extra}"
fi

scanned=0
violations=0
while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  scanned=$((scanned + 1))
  [[ "${1:-}" == "--list" ]] && { echo "$f"; continue; }
  [[ "$f" =~ $ALLOW ]] && continue
  if out=$(grep -InE "$PAT" "$f" 2>/dev/null); then
    printf '\n\033[31m✗ %s\033[0m\n' "$f"
    echo "$out" | head -6 | sed 's/^/    /'
    violations=$((violations + 1))
  fi
done < <(git ls-files; git ls-files --others --exclude-standard)

[[ "${1:-}" == "--list" ]] && exit 0
echo
echo "————————————————————————————————"
if [[ $violations -eq 0 ]]; then
  printf '\033[32mPASS\033[0m  %d files scanned, no leaks\n' "$scanned"
  exit 0
fi
printf '\033[31mFAIL\033[0m  %d files scanned, %d with leaks\n' "$scanned" "$violations"
echo
echo "Fixes: absolute paths -> \"\${RCDL_X:-/path/to/...}\" with the real value in scripts/local.env;"
echo "       machine names -> \"the board\" / \"the convert host\"; private notes -> docs/private/"
exit 1
