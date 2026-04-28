#!/usr/bin/env bash
# Start all position-specific Free Agency GM APIs (ports 8002–8013).
# Run from anywhere; uses repo root on PYTHONPATH.
# Ctrl+C stops all child uvicorn processes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"

PIDS=()

cleanup() {
  for p in "${PIDS[@]}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start() {
  local module=$1
  local port=$2
  echo "[FA APIs] Starting ${module} on port ${port}"
  uvicorn "${module}:app" --host 127.0.0.1 --port "${port}" &
  PIDS+=($!)
}

start backend.agent.ed_main_api 8002
start backend.agent.di_main_api 8003
start backend.agent.qb_main_api 8004
start backend.agent.hb_main_api 8005
start backend.agent.wr_main_api 8006
start backend.agent.te_main_api 8007
start backend.agent.t_main_api 8008
start backend.agent.g_main_api 8009
start backend.agent.c_main_api 8010
start backend.agent.lb_main_api 8011
start backend.agent.cb_main_api 8012
start backend.agent.s_main_api 8013

echo "[FA APIs] 12 servers running. Ctrl+C to stop all."
wait
