"""Push the cached teacher outputs to a public Hub dataset, resumably.

`upload_large_folder` keeps its own state under <folder>/.cache/huggingface, so a
killed process resumes where it stopped instead of re-sending 300 GB.  It is
called in a retry loop anyway: a multi-hour transfer will meet a transient error,
and the point of running detached is that nobody is watching when it does.
"""
import os
import sys
import time

from huggingface_hub import HfApi

REPO = os.environ.get('HF_REPO', 'rudals/resmap-navsim-teacher-kd')
FOLDER = os.environ.get('CACHE_DIR', '/data3/kyungmin/kd_teacher_resmap')
TOKEN = os.environ['HF_TOKEN']

api = HfApi(token=TOKEN)
api.create_repo(REPO, repo_type='dataset', private=False, exist_ok=True)
print(f'[upload] repo ready: https://huggingface.co/datasets/{REPO}', flush=True)
print(f'[upload] folder: {FOLDER}', flush=True)

attempt = 0
while True:
    attempt += 1
    try:
        api.upload_large_folder(
            repo_id=REPO,
            repo_type='dataset',
            folder_path=FOLDER,
            num_workers=8,
            print_report=True,
        )
        print(f'[upload] complete after {attempt} attempt(s)', flush=True)
        break
    except Exception as exc:                       # noqa: BLE001
        wait = min(300, 30 * attempt)
        print(f'[upload] attempt {attempt} failed: {type(exc).__name__}: '
              f'{exc}\n[upload] retrying in {wait}s', flush=True)
        time.sleep(wait)

files = api.list_repo_files(REPO, repo_type='dataset')
print(f'[upload] repo now holds {len(files)} files', flush=True)
