#!/usr/bin/env bash
# Records a demo session with asciinema and renders it with agg.
#   docs/demo/record.sh <session> [--gif out.gif] [--still out.png]
# <session> is the name of docs/demo/session_<name>.sh (tiers, fallback, ecs, monitoring, tests).
# The recording is a REAL terminal session: commands are typed by a script, then actually run.
set -euo pipefail
cd "$(dirname "$0")/../.."
name="$1"; shift
cast="docs/demo/casts/${name}.cast"
cols=${COLS:-118}; rows=${ROWS:-30}
# run asciinema inside a pty of the requested size, so the recorded terminal really is cols x rows
python3 docs/demo/pty_run.py "$cols" "$rows" -- asciinema rec --overwrite --quiet --title "hybrid-risk: ${name}" \
  --command "bash docs/demo/session_${name}.sh" "$cast" > /dev/null
echo "recorded $cast"
