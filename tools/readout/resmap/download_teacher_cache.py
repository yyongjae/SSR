"""Fetch the public ReSMap KD cache from the Hugging Face Hub (resumable).

    python tools/readout/resmap/download_teacher_cache.py --out /data/kd_teacher_resmap
    # BEV only (what readout training and distillation read), ~301 GB:
    python tools/readout/resmap/download_teacher_cache.py --out ... --fields bev
    # add vectors/scores/labels for map_label_source=teacher (~0.1 GB):
    python tools/readout/resmap/download_teacher_cache.py --out ... --fields bev vectors scores labels
    # navtest teacher features only (PDMS of readouts, ~31 GB) -> <out>/navtest
    python tools/readout/resmap/download_teacher_cache.py --out ... --subsets navtest --fields bev

--subsets picks the navtrain cache (repo root) and/or navtest/ (same format).

Re-running continues an interrupted download.  HF_HUB_ENABLE_HF_TRANSFER=1
(pip install hf_transfer) is much faster on a fat pipe.
"""
import argparse

from huggingface_hub import snapshot_download

REPO = "rudals/resmap-navsim-teacher-kd"
FIELDS = ("bev", "seg", "vectors", "scores", "labels", "props")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--fields", nargs="+", default=list(FIELDS), choices=FIELDS)
    ap.add_argument("--subsets", nargs="+", default=["train"], choices=["train", "navtest"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    patterns = ["README.md"]
    for subset in args.subsets:
        prefix = "" if subset == "train" else f"{subset}/"
        patterns += [f"{prefix}index.json", f"{prefix}meta.json"] + [f"{prefix}{f}/*" for f in args.fields]
    path = snapshot_download(
        repo_id=args.repo, repo_type="dataset", local_dir=args.out,
        allow_patterns=patterns, max_workers=args.workers,
    )
    print(f"cache at {path}")


if __name__ == "__main__":
    main()
