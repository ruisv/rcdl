#!/usr/bin/env bash
# Install the repo's git hooks (pre-commit leak scan). Hooks live in .git/ and
# are never committed, so run this once per clone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
hook=.git/hooks/pre-commit
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# RCDL pre-commit: refuse to commit private material.
exec scripts/check_publishable.sh
HOOK
chmod +x "$hook"
# commit-msg: reject tool-generated trailers / session links in the message.
cat > .git/hooks/commit-msg <<'HOOK'
#!/usr/bin/env bash
PAT='Co-Authored-By:.*(noreply@|\[bot\])|[A-Za-z]+-Session: *https?://|[Gg]enerated with \[?[A-Z][a-z]+ [A-Z]'
PRIV="local/private_patterns.txt"
if [ -f "$PRIV" ]; then
  extra="$(grep -vE '^\s*(#|$)' "$PRIV" | paste -sd'|' -)"
  [ -n "$extra" ] && PAT="${PAT}|${extra}"
fi
if grep -qiE "$PAT" "$1"; then
  echo "commit-msg: the message contains a tool trailer / private reference — remove it" >&2
  exit 1
fi
HOOK
chmod +x .git/hooks/commit-msg
echo ">> installed .git/hooks/pre-commit + commit-msg"
