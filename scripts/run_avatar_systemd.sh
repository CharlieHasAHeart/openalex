#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

RUNS_DIR="${RUNS_DIR:-runs}"
WORKERS="${WORKERS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
AUTHOR_LIMIT="${AUTHOR_LIMIT:-0}"
AUTHOR_OFFSET="${AUTHOR_OFFSET:-0}"
AUTHOR_IDS_FILE="${AUTHOR_IDS_FILE:-}"

RESUME_RUN_ID="$(
python3 - "${RUNS_DIR}" <<'PY'
import glob
import json
import os
import sys

runs_dir = sys.argv[1]
summaries = glob.glob(os.path.join(runs_dir, "*", "*", "summary.json"))
rows = []
for path in summaries:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    if data.get("run_status") != "running":
        continue
    run_id = str(data.get("run_id") or "").strip()
    ts = str(data.get("last_updated_at") or data.get("started_at") or "")
    if run_id:
        rows.append((ts, run_id))
rows.sort()
if rows:
    print(rows[-1][1])
PY
)"

CMD=(python3 main.py --workers "${WORKERS}" --progress-every "${PROGRESS_EVERY}" --log-level "${LOG_LEVEL}" --runs-dir "${RUNS_DIR}")

if [[ -n "${AUTHOR_IDS_FILE}" ]]; then
  CMD+=(--author-ids-file "${AUTHOR_IDS_FILE}")
else
  CMD+=(--author-limit "${AUTHOR_LIMIT}" --author-offset "${AUTHOR_OFFSET}")
fi

if [[ -n "${RESUME_RUN_ID}" ]]; then
  CMD+=(--resume-run-id "${RESUME_RUN_ID}")
fi

exec "${CMD[@]}"

