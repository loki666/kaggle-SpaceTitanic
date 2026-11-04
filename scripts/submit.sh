#!/usr/bin/env bash
# Usage: scripts/submit.sh submissions/submission.csv "message"
set -euo pipefail
FILE="${1:-submissions/submission.csv}"
MSG="${2:-ensemble}"
kaggle competitions submit -c spaceship-titanic -f "$FILE" -m "$MSG"
