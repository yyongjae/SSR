#!/usr/bin/env bash
# OPTIONAL, research tools only: the navtest metric cache built by this repo's navsim fork, at SSR_NAVTEST_MC.  3 GiB,
# 57 min at 12 workers on the reference server.  Nothing on the release path reads it: the training cache's saved config is
# vendored (resources/metric_cache_saved_config.yaml), and reported numbers use the pristine v1.1 cache.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
refuse_unless_resume "$SSR_NAVTEST_MC"
need "$SSR_TEST_DATA_ROOT/navsim_logs/test/test"
run_mkdir "$NAVSIM_EXP_ROOT"
cd "$REPO"
run "${CPU[@]}" "$SSR_PY" navsim/planning/script/run_metric_caching.py worker=single_machine_thread_pool worker.max_workers=12 \
  scene_filter=navtest split=test navsim_log_path="$SSR_TEST_DATA_ROOT/navsim_logs/test/test" \
  cache.cache_path="$SSR_NAVTEST_MC" worker.use_process_pool=true
