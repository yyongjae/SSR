#!/bin/bash
# Alias of tools/ck/smoke_ck.sh (task-list name): <= 200 tokens, dump -> eval end to end on GPU 3.
exec bash "$(dirname "${BASH_SOURCE[0]}")/smoke_ck.sh" "$@"
