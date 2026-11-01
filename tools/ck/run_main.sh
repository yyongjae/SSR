#!/bin/bash
# Alias of tools/ck/run_phase1.sh (task-list name). Same arguments: <stage>|from:<stage>|status [--bg]
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_phase1.sh" "$@"
