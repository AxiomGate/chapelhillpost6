#!/usr/bin/env bash
# Overnight half of the daily run: research and draft, then stop.
#
# The draft is waiting when you sit down; you review and approve, and
# `podcastpipe finish` does the rest. Wire this to cron or a systemd timer:
#
#   crontab -e
#   30 4 * * * /opt/podcastpipe/scripts/daily.sh >> /home/you/podcast-cron.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

source .venv/bin/activate

echo "=== $(date -Iseconds) starting daily research ==="
podcastpipe research
podcastpipe script

# Keep two weeks of intermediates; masters and metadata are always kept.
podcastpipe gc --days 14

echo "=== $(date -Iseconds) draft ready for review ==="
podcastpipe status --limit 5
