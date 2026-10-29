"""scripts/km_corridor/v6c_pipeline.sh (CPU, stubbed, report 129 F09): the pipeline runs from a throw-away git copy of
scripts/km_corridor (v6c_pipeline.sh, env.example.sh, train.sh) with PY = a bash stub that only logs its argv, and a
fake nvidia-smi first on PATH, so no Python, GPU or training starts.  Checks the stage order and arguments (--phase
check first, --seed on every train_teachers call, GPUs 0-3, EXACT2_DIR exported), resume, the code_rev / legacy-dir /
running-pipeline guards (a refused launch leaves the dir untouched), the GPU preflight, the label-table provenance
(re-listed at every launch once a stage is done, and before the student) and STUDENT_RESUME (only with its own
teacher)."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "km_corridor"
TABLES = ("corridor64", "exact", "exact2")
SPLITS = ("navtrain_train", "navtrain_val")
STAGES_T = ("check", "fields", "platt", "te", "finish")
# the parent environment must not leak into the stubbed runs
DROP = ("GPUS", "GPU_CHECK", "GPU_MAX_USED_MIB", "ALLOW_REUSE", "COVERAGE_MIN", "SEED_T", "STUDENT", "STUDENT_RESUME",
        "EXACT2_DIR", "KM_ROOT", "KM_RUN_ROOT", "PY", "CUDA_VISIBLE_DEVICES", "WANDB", "WANDB_GROUP", "DRYRUN",
        "EPOCHS_T", "BATCH_T", "LIMIT_T", "PLATT_BATCHES", "LR_T", "WORKERS_T", "WORKERS_TE", "EPOCHS", "LIMIT_TRAIN",
        "LIMIT_VAL", "VAL_EVERY", "LR", "MAXEP", "SELECT_RULE", "MAP_REPR", "DISTILL", "C0_CKPT", "STUB_CHECK_RC",
        "STUB_READ_RULE", "STUB_RR_FAIL", "STUB_TRAIN_RC", "STUB_ON_FINISH", "SMI_BUSY_GPU", "SMI_BUSY_MIB",
        "SMI_IDLE_MIB", "SMI_RC")

STUB_PY = r"""#!/usr/bin/env bash
# stub python: one log line per call (newlines of a -c snippet folded), never runs anything
a="$*"; a=${a//$'\n'/ }
echo "CVD=${CUDA_VISIBLE_DEVICES-unset} EX2=${EXACT2_DIR-unset} $a" >> "$STUB_LOG"
case "$a" in
  *READ_RULE*) [ -n "${STUB_RR_FAIL:-}" ] && { echo "ImportError: cannot import name 'READ_RULE'" >&2; exit 1; }
               echo "${STUB_READ_RULE-r129}"; exit 0 ;;
  *"--phase check"*) exit ${STUB_CHECK_RC:-0} ;;
  # a table rebuilt in place while stage T runs: rewrite the meta.txt named by STUB_ON_FINISH
  *"--phase finish"*) [ -n "${STUB_ON_FINISH:-}" ] && echo "rebuilt during stage T" > "$STUB_ON_FINISH"; exit 0 ;;
  *run_training.py*) exit ${STUB_TRAIN_RC:-0} ;;           # STUB_TRAIN_RC=1: an interrupted student
esac
exit 0
"""
FAKE_SMI = r"""#!/usr/bin/env bash
echo "$*" >> "$SMI_LOG"
[ -n "${SMI_RC:-}" ] && { echo "NVIDIA-SMI has failed"; exit $SMI_RC; }
i=""; while [ $# -gt 0 ]; do [ "$1" = -i ] && i=$2; shift; done
if [ "$i" = "${SMI_BUSY_GPU:-}" ] || [ "${SMI_BUSY_GPU:-}" = all ]; then echo "${SMI_BUSY_MIB:-5000}"; else echo "${SMI_IDLE_MIB:-3}"; fi
"""


def _exe(p: Path, text: str) -> Path:
    p.write_text(text)
    p.chmod(0o755)
    return p


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class Pipe:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.repo = tmp / "repo"
        (self.repo / "scripts" / "km_corridor").mkdir(parents=True)
        for f in ("v6c_pipeline.sh", "env.example.sh", "train.sh"):
            shutil.copy(SCRIPTS / f, self.repo / "scripts" / "km_corridor" / f)
        _git(self.repo, "init", "-q")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "init")
        self.km = tmp / "km"
        for t in TABLES:
            for s in SPLITS:
                d = self.km / t / s
                d.mkdir(parents=True)
                if (t, s) != ("corridor64", "navtrain_val"):          # one table without meta.txt / *.npy
                    (d / "meta.txt").write_text(f"split {s} table {t} v1\n")
                    (d / "agent_rows.npy").write_bytes(b"rows" * 4)
                    os.utime(d / "agent_rows.npy", (1.7e9, 1.7e9))
        self.runs = tmp / "runs"
        self.bin = tmp / "bin"
        self.bin.mkdir()
        _exe(self.bin / "nvidia-smi", FAKE_SMI)
        self.py = _exe(tmp / "stub_py", STUB_PY)
        self.log = tmp / "stub.log"
        self.smi_log = tmp / "smi.log"

    def run(self, *args: str, unset: tuple = (), **env) -> subprocess.CompletedProcess:
        e = {k: v for k, v in os.environ.items() if k not in DROP}
        e.update(PATH=f"{self.bin}:{e['PATH']}", PY=str(self.py), KM_ROOT=str(self.km), KM_RUN_ROOT=str(self.runs),
                 GPUS="0,1,2,3", GPU_CHECK="0", WANDB="false", STUDENT="0", STUB_LOG=str(self.log),
                 SMI_LOG=str(self.smi_log))
        e.update({k: str(v) for k, v in env.items()})
        for k in unset:
            e.pop(k, None)
        return subprocess.run(["bash", str(self.repo / "scripts" / "km_corridor" / "v6c_pipeline.sh"), *args],
                              env=e, capture_output=True, text=True, timeout=120)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def work_calls(self) -> list[str]:   # everything but the READ_RULE probe of the code_rev guard
        return [c for c in self.calls() if "READ_RULE" not in c]

    def out(self, name: str) -> Path:
        return self.runs / "v6c_teacher" / name

    def snap(self, name: str) -> dict:   # every file (and dir) of OUT with its bytes
        out = self.out(name)
        return {str(p.relative_to(out)): (p.read_bytes() if p.is_file() else None) for p in sorted(out.rglob("*"))}


@pytest.fixture()
def pipe(tmp_path):
    return Pipe(tmp_path)


def _phase(c: str) -> str:
    return c.split("--phase ")[1].split()[0] if "--phase " in c else ""


def _rev(repo: Path) -> str:
    return _git(repo, "rev-parse", "--short=12", "HEAD")


def _npy_md5(d: Path) -> str:   # the pipeline's 'stat -Lc "%n %s %Y" *.npy | md5sum' of one table dir
    st = "".join(f"{p.name} {p.stat().st_size} {int(p.stat().st_mtime)}\n" for p in sorted(d.glob("*.npy")))
    return hashlib.md5(st.encode()).hexdigest()


def test_syntax():
    subprocess.run(["bash", "-n", str(SCRIPTS / "v6c_pipeline.sh")], check=True)


def test_stage_t_order_args_and_files(pipe):
    r = pipe.run("tst")
    assert r.returncode == 0, r.stdout + r.stderr
    out = pipe.out("tst")
    for st in STAGES_T:
        assert (out / f"done_{st}").is_file(), st
    assert not (out / "done_student").exists()                                  # STUDENT=0
    calls = pipe.calls()
    assert "READ_RULE" in calls[0]                                              # the code_rev probe comes first
    tt = [c for c in calls if "train_teachers.py" in c]
    assert "--phase check --min-coverage 1.0" in tt[0] and tt[0].startswith("CVD= ")   # CPU only
    assert sorted(_phase(c) for c in tt[1:5]) == ["td", "td", "tm", "tm"]                # 4 parallel copies
    assert [_phase(c) for c in tt[5:]] == ["merge_fields", "platt_occ", "te_fold", "te_fold", "finish"]
    assert all("--seed 0" in c for c in tt)
    # default teacher hyper-parameters are the 10-28 recipe
    assert all("--epochs 10 --batch 8 --limit 0 --platt-batches 200 --lr 2e-4 --seed 0" in c for c in tt)
    assert all("--workers 16" in c for c in tt if _phase(c) in ("check", "td", "tm", "merge_fields", "platt_occ"))
    assert all("--workers 8" in c for c in tt if _phase(c) in ("te_fold", "finish"))
    # GPUs 0-3 only, and every call sees the exported EXACT2_DIR
    cvd = {c.split()[0][4:] for c in calls}
    assert cvd <= {"", "unset", "0", "1", "2", "3", "0,1", "2,3"}, cvd
    assert all(f"EX2={pipe.km}/exact2 " in c for c in calls)
    assert sum("load_teachers" in c for c in calls) == 1
    # code_rev = 12-char git rev of the (clean) copy + READ_RULE
    rev = _rev(pipe.repo)
    assert len(rev) == 12 and (out / "code_rev").read_text().strip() == f"{rev} read_rule=r129"
    prov = (out / "tables_provenance.txt").read_text().splitlines()
    assert len(prov) == 7 and prov[-1] == f"git {rev} read_rule=r129"
    rows = {tuple(ln.split()[:2]): ln.split()[2:] for ln in prov[:-1]}
    assert set(rows) == {(t, s) for t in TABLES for s in SPLITS}
    d = pipe.km / "exact2" / "navtrain_val"
    md5 = hashlib.md5((d / "meta.txt").read_bytes()).hexdigest()
    assert rows[("exact2", "navtrain_val")] == [os.path.realpath(d), md5, _npy_md5(d)]
    assert rows[("corridor64", "navtrain_val")][1:] == ["no-meta", "no-npy"]
    st = (out / "status.log").read_text()
    assert "GPUs 0 1 2 3" in st and f"EXACT2_DIR {pipe.km}/exact2 -> {os.path.realpath(pipe.km / 'exact2')}" in st


def test_provenance_reads_exact2_current(pipe):
    # a v2 split filled by merge_exact_parts.py is read through <split>/current -> <split>/v<time>
    sp = pipe.km / "exact2" / "navtrain_train"
    (sp / "v2").mkdir()
    (sp / "v2" / "meta.txt").write_text("merged parts p0 p1\n")
    (sp / "v2" / "agent_rows.npy").write_bytes(b"merged")
    (sp / "current").symlink_to("v2")
    assert pipe.run("tst").returncode == 0
    (row,) = [ln.split() for ln in (pipe.out("tst") / "tables_provenance.txt").read_text().splitlines()
              if ln.startswith("exact2 navtrain_train ")]
    assert row[2:] == [os.path.realpath(sp / "v2"), hashlib.md5(b"merged parts p0 p1\n").hexdigest(),
                       _npy_md5(sp / "v2")]


def test_default_gpus_and_status(pipe):
    r = pipe.run("tst", unset=("GPUS", "GPU_CHECK"))                            # defaults: GPUs 0-3, preflight on
    assert r.returncode == 0, r.stdout + r.stderr
    q = pipe.smi_log.read_text().splitlines()
    assert q == [f"--query-gpu=memory.used --format=csv,noheader,nounits -i {g}" for g in range(4)]
    assert {c.split()[0] for c in pipe.calls()} <= {"CVD=", "CVD=unset", "CVD=0", "CVD=1", "CVD=2", "CVD=3",
                                                    "CVD=0,1", "CVD=2,3"}
    pipe.smi_log.unlink()
    n = len(pipe.calls())
    s = pipe.run("tst", "status", unset=("GPU_CHECK",), SMI_BUSY_GPU="all")    # status: no GPU check, no python
    assert s.returncode == 0, s.stdout + s.stderr
    assert not pipe.smi_log.exists() and len(pipe.calls()) == n
    lines = s.stdout.splitlines()
    assert any(ln.startswith("code_rev ") for ln in lines)
    for st in STAGES_T:
        assert any(ln.split()[:2] == [st, "done"] for ln in lines), st
    assert any(ln.split()[:2] == ["student", "pending"] for ln in lines)
    assert pipe.run("nope", "status").returncode == 0 and not pipe.out("nope").exists()   # read-only


def test_rerun_resumes_without_new_work(pipe):
    assert pipe.run("tst").returncode == 0
    n = len(pipe.work_calls())
    r = pipe.run("tst")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(pipe.work_calls()) == n                                          # every stage skipped


def test_code_rev_guard(pipe):
    assert pipe.run("tst").returncode == 0
    out = pipe.out("tst")
    n = len(pipe.work_calls())
    (out / "code_rev").write_text("0000000 read_rule=legacy\n")
    before = pipe.snap("tst")
    r = pipe.run("tst")
    assert r.returncode == 2 and "OUT was produced by 0000000 read_rule=legacy, code is" in r.stdout
    assert len(pipe.work_calls()) == n and pipe.snap("tst") == before          # refused: OUT untouched
    r = pipe.run("tst", ALLOW_REUSE=1)
    assert r.returncode == 0 and "WARNING: ALLOW_REUSE=1" in r.stdout
    assert "WARNING: ALLOW_REUSE=1" in (out / "status.log").read_text()
    assert (out / "code_rev").read_text() == "0000000 read_rule=legacy\n"      # not re-stamped
    # a dirty tree, another READ_RULE and a new commit are all a different code_rev (fresh name 'a')
    env_sh = pipe.repo / "scripts" / "km_corridor" / "env.example.sh"
    assert pipe.run("a").returncode == 0
    env_sh.write_text((SCRIPTS / "env.example.sh").read_text() + "\n# edit\n")
    r = pipe.run("a")
    assert r.returncode == 2 and re.search(r"code is [0-9a-f]{12}-dirty\.[0-9a-f]{8} read_rule=r129", r.stdout)
    env_sh.write_text((SCRIPTS / "env.example.sh").read_text())                 # touched but unchanged: clean
    assert pipe.run("a").returncode == 0
    assert pipe.run("a", STUB_READ_RULE="r130").returncode == 2
    (pipe.repo / "x.txt").write_text("x")
    _git(pipe.repo, "add", "x.txt")
    _git(pipe.repo, "commit", "-qm", "x")
    assert pipe.run("a").returncode == 2


def test_code_rev_dirty_content_and_untracked_code(pipe):
    env_sh = pipe.repo / "scripts" / "km_corridor" / "env.example.sh"
    env_sh.write_text(env_sh.read_text() + "\n# edit 1\n")
    assert pipe.run("z").returncode == 0
    c1 = (pipe.out("z") / "code_rev").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{12}-dirty\.[0-9a-f]{8} read_rule=r129", c1), c1
    assert pipe.run("z").returncode == 0                                        # the same dirty tree resumes
    env_sh.write_text(env_sh.read_text() + "\n# edit 2 (other code)\n")
    r = pipe.run("z")                                                           # another dirty tree: refused
    assert r.returncode == 2 and f"OUT was produced by {c1}, code is" in r.stdout
    # a new, still-untracked module under navsim / tools / scripts is dirty; an untracked test or note is not
    _git(pipe.repo, "checkout", "-q", "--", ".")
    assert pipe.run("u").returncode == 0
    (pipe.repo / "tests").mkdir()
    (pipe.repo / "tests" / "test_x.py").write_text("x = 1\n")
    (pipe.repo / "notes.md").write_text("n\n")
    assert pipe.run("u").returncode == 0
    (pipe.repo / "tools").mkdir()
    (pipe.repo / "tools" / "new_mod.py").write_text("x = 1\n")
    r = pipe.run("u")
    assert r.returncode == 2 and "-dirty." in r.stdout
    (pipe.repo / "tools" / "new_mod.py").write_text("x = 2\n")                  # its content is in the hash
    assert pipe.run("u2").returncode == 0
    (pipe.repo / "tools" / "new_mod.py").write_text("x = 3\n")
    assert pipe.run("u2").returncode == 2


@pytest.mark.parametrize("layout", ["main", "platt3", "platt2", "part_ckpt", "coverage"])
def test_legacy_dir_aborts(pipe, layout):
    # pre-r129 dirs (read-only listing 10-29): v6cT_main / smoke have done markers; v6cT_platt2 / platt3 only teachers,
    # parts and logs (train_teachers runs outside the pipeline)
    files = {"main": ["done_fields", "done_platt", "teachers_te.pt", "part_td0.ckpt", "log_td_f0.jsonl", "pipeline.pid"],
             "platt3": ["log_finish.jsonl", "part_te0.pt", "part_te1.pt", "teachers.pt", "teachers_platt_occ.pt",
                        "teachers_te.pt", "logs/te0.log"],
             "platt2": ["log_finish.jsonl", "log_platt_occ.jsonl", "part_te0.pt", "part_te1.pt", "teachers.pt",
                        "teachers_fields.pt", "teachers_platt_occ.pt", "teachers_te.pt"],
             "part_ckpt": ["part_tm1.ckpt"], "coverage": ["coverage.json"]}[layout]
    out = pipe.out("v6cT_legacy")
    for f in files:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_bytes(b"legacy " + f.encode())
    if layout == "main":
        (out / "pipeline.pid").write_text("999999999\n")                         # dead
    before = pipe.snap("v6cT_legacy")
    r = pipe.run("v6cT_legacy")
    assert r.returncode == 2 and "pre-r129" in r.stdout and "use a new name or ALLOW_REUSE=1" in r.stdout
    assert pipe.work_calls() == [] and pipe.snap("v6cT_legacy") == before       # no code_rev, no log, no status.log
    if layout == "main":
        r = pipe.run("v6cT_legacy", ALLOW_REUSE=1)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not (out / "code_rev").exists()                                  # never stamped with the new rev
        assert not any(_phase(c) in ("td", "tm") for c in pipe.calls())         # done_fields kept


def test_fresh_dir_with_only_logs_is_not_legacy(pipe):
    out = pipe.out("tst")                                                       # e.g. a launch refused by older code
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "read_rule.log").write_text("")
    (out / "status.log").write_text("[..] abort: GPU 0 already uses 5000 MiB\n")
    r = pipe.run("tst")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "code_rev").is_file() and (out / "done_finish").is_file()


def test_read_rule_missing_aborts(pipe):
    r = pipe.run("tst", STUB_READ_RULE="")
    assert r.returncode == 2 and "READ_RULE missing" in r.stdout
    assert pipe.work_calls() == [] and not pipe.out("tst").exists()            # nothing written
    r = pipe.run("tst", STUB_RR_FAIL=1)
    assert r.returncode == 2 and "READ_RULE missing" in r.stdout and "ImportError" in r.stdout
    assert not pipe.out("tst").exists()


def test_running_pipeline_refused(pipe, tmp_path):
    assert pipe.run("tst").returncode == 0
    out = pipe.out("tst")
    (tmp_path / "fake").mkdir()
    fake = _exe(tmp_path / "fake" / "v6c_pipeline.sh", "while :; do sleep 0.2; done\n")
    procs = []
    try:
        for name in ("tst", "other"):                 # a live 'v6c_pipeline.sh tst', and one on another name
            procs.append(subprocess.Popen(["bash", str(fake), name]))
        time.sleep(0.3)
        n = len(pipe.work_calls())
        (out / "done_finish").unlink()
        (out / "pipeline.pid").write_text(f"{procs[0].pid}\n")
        before = pipe.snap("tst")
        r = pipe.run("tst")
        assert r.returncode == 2 and f"pipeline pid {procs[0].pid} still runs" in r.stdout
        assert len(pipe.work_calls()) == n and pipe.snap("tst") == before
        r = pipe.run("tst", ALLOW_REUSE=1)
        assert r.returncode == 0 and f"WARNING: ALLOW_REUSE=1, pipeline pid {procs[0].pid} still runs" in r.stdout
        (out / "done_finish").unlink()
        (out / "pipeline.pid").write_text(f"{procs[1].pid}\n")                  # another name's pipeline
        assert pipe.run("tst").returncode == 0
    finally:
        for p in procs:
            p.kill()
            p.wait()
    (out / "done_finish").unlink()
    (out / "pipeline.pid").write_text(f"{procs[0].pid}\n")                      # dead now
    assert pipe.run("tst").returncode == 0


def test_check_stage_failure_stops(pipe):
    r = pipe.run("tst", STUB_CHECK_RC=2)
    assert r.returncode == 1 and "stage check FAILED" in r.stdout
    assert not (pipe.out("tst") / "done_check").exists()
    assert [_phase(c) for c in pipe.work_calls()] == ["check"]                  # nothing after stage check


def test_tables_rebuilt_after_failed_check(pipe):
    # stage check fails, the tables are rebuilt, the same command re-runs: the list is re-written before stage check,
    # so stage T and the student see one table set and the student starts
    assert pipe.run("tst", STUB_CHECK_RC=2).returncode == 1
    meta = pipe.km / "exact2" / "navtrain_train" / "meta.txt"
    meta.write_text("split navtrain_train table exact2 ok 85109/85109 rebuilt\n")
    r = pipe.run("tst")
    assert r.returncode == 0, r.stdout + r.stderr
    prov = (pipe.out("tst") / "tables_provenance.txt").read_text()
    assert hashlib.md5(meta.read_bytes()).hexdigest() in prov
    r = pipe.run("tst", STUDENT=1)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(_student_calls(pipe)) == 1


@pytest.mark.parametrize("busy", ["0", "2", "all"])
def test_gpu_preflight_aborts(pipe, busy):
    r = pipe.run("tst", unset=("GPU_CHECK",), SMI_BUSY_GPU=busy)
    g = "0" if busy == "all" else busy
    assert r.returncode == 2 and f"GPU {g} already uses 5000 MiB" in r.stdout
    assert pipe.calls() == []                                                   # before any python call
    assert not pipe.out("tst").exists()                                         # nothing written


def test_gpu_preflight_threshold_and_errors(pipe):
    assert pipe.run("tst", unset=("GPU_CHECK",), SMI_BUSY_GPU="1", GPU_MAX_USED_MIB=6000).returncode == 0
    r = pipe.run("tst2", unset=("GPU_CHECK",), SMI_RC=9)                       # unreadable nvidia-smi: abort, not pass
    assert r.returncode == 2 and "cannot read the memory use of GPU 0" in r.stdout
    assert pipe.run("tst2", SMI_RC=9).returncode == 0                           # GPU_CHECK=0 skips it


def _student_calls(pipe):
    return [c for c in pipe.calls() if "run_training.py" in c]


def _student_prov(pipe, name: str) -> str:   # what a student run dir records: stage T's list + the teacher
    out = pipe.out(name)
    return (out / "tables_provenance.txt").read_text() + f"teacher {out}/teachers_te.pt\n"


def test_student_fresh_args_and_provenance_copy(pipe):
    r = pipe.run("tst", STUDENT=1)
    assert r.returncode == 0, r.stdout + r.stderr
    out = pipe.out("tst")
    (st,) = _student_calls(pipe)
    assert st.startswith("CVD=0,1,2,3 ") and "resume_checkpoint" not in st
    assert f"++agent.config.v6c_teacher_ckpt={out}/teachers_te.pt" in st and "wandb.group=refiner-v6c" in st
    assert f"++agent.config.exact2_dir={pipe.km}/exact2 " in st
    assert "dataloader.params.batch_size=4" in st and "trainer.params.accumulate_grad_batches=8" in st
    assert (out / "done_student").is_file()
    assert (pipe.runs / "corr30" / "tst_student" / "tables_provenance.txt").read_text() == _student_prov(pipe, "tst")


def _ckpts(pipe, student: str):
    sd = pipe.runs / "corr30" / student / "train" / "lightning_logs"
    old = sd / "version_0" / "checkpoints" / "last.ckpt"
    new = sd / "version_1" / "checkpoints" / "last.ckpt"
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"ckpt")
    os.utime(old, (1e9, 1e9))
    return old, new


def test_student_resume(pipe):
    assert pipe.run("tst").returncode == 0
    r = pipe.run("tst", STUDENT=1, STUDENT_RESUME=1)                           # nothing to resume from
    assert r.returncode == 1 and "no last.ckpt to resume" in r.stdout and _student_calls(pipe) == []
    r = pipe.run("tst", STUDENT=1, STUB_TRAIN_RC=1)                             # an interrupted fresh student
    assert r.returncode == 1 and "stage student FAILED" in r.stdout and len(_student_calls(pipe)) == 1
    old, new = _ckpts(pipe, "tst_student")
    r = pipe.run("tst", STUDENT=1)                                              # checkpoints, no STUDENT_RESUME
    assert r.returncode == 1 and "already has checkpoints" in r.stdout and len(_student_calls(pipe)) == 1
    r = pipe.run("tst", STUDENT=1, STUDENT_RESUME=1)
    assert r.returncode == 0, r.stdout + r.stderr
    st = _student_calls(pipe)[-1]
    assert f"++resume_checkpoint='{new}'" in st and str(old) not in st          # the newest last.ckpt
    assert f"resumes from {new}" in (pipe.out("tst") / "status.log").read_text()


def test_student_resume_needs_its_own_teacher(pipe):
    # student S started with teacher pipeline A, then resumed from pipeline B: refused, S keeps A's record
    r = pipe.run("A", "S", STUDENT=1, STUB_TRAIN_RC=1)
    assert r.returncode == 1 and len(_student_calls(pipe)) == 1
    _ckpts(pipe, "S")
    rec = pipe.runs / "corr30" / "S" / "tables_provenance.txt"
    assert rec.read_text() == _student_prov(pipe, "A")
    r = pipe.run("B", "S", STUDENT=1, STUDENT_RESUME=1)
    assert r.returncode == 1 and "names another teacher" in r.stdout, r.stdout + r.stderr
    assert len(_student_calls(pipe)) == 1 and rec.read_text() == _student_prov(pipe, "A")
    assert (pipe.out("B") / "logs" / "student_provenance_diff.log").read_text().strip()
    rec.unlink()                                                                # no record: cannot be checked
    r = pipe.run("A", "S", STUDENT=1, STUDENT_RESUME=1)
    assert r.returncode == 1 and "is missing" in r.stdout and len(_student_calls(pipe)) == 1
    rec.write_text(_student_prov(pipe, "A"))
    r = pipe.run("A", "S", STUDENT=1, STUDENT_RESUME=1)                         # its own teacher: resumes
    assert r.returncode == 0, r.stdout + r.stderr
    assert "++resume_checkpoint=" in _student_calls(pipe)[-1]


@pytest.mark.parametrize("change", ["meta", "npy_mtime", "current", "exact2_dir", "stored_line"])
@pytest.mark.parametrize("when", ["student", "stage_t_resume"])
def test_tables_change_between_launches_aborts(pipe, change, when):
    assert pipe.run("tst").returncode == 0
    out = pipe.out("tst")
    if when == "stage_t_resume":                                                # stage T interrupted after stage te
        for st in ("te", "finish"):
            (out / f"done_{st}").unlink()
    env = {}
    if change == "meta":                                                        # exact2 rebuilt after the done stages
        (pipe.km / "exact2" / "navtrain_val" / "meta.txt").write_text("split navtrain_val table exact2 v2\n")
    elif change == "npy_mtime":                                                 # rebuilt in place, same meta text
        os.utime(pipe.km / "exact" / "navtrain_train" / "agent_rows.npy", (1.8e9, 1.8e9))
    elif change == "current":                                                   # a merged v2 version swapped in
        sp = pipe.km / "exact2" / "navtrain_train"
        shutil.copytree(sp, pipe.tmp / "v_new")
        (pipe.tmp / "v_new" / "meta.txt").write_text("merged parts p0\n")
        (sp / "current").symlink_to(pipe.tmp / "v_new")
    elif change == "exact2_dir":                                                # this launch points elsewhere
        shutil.copytree(pipe.km / "exact2", pipe.tmp / "exact2_other")
        env["EXACT2_DIR"] = pipe.tmp / "exact2_other"
    else:
        p = out / "tables_provenance.txt"                                     # one stored table line edited
        p.write_text("".join(ln.rsplit(" ", 1)[0] + " 0123abcd\n" if ln.startswith("exact navtrain_train ") else ln
                             for ln in p.read_text().splitlines(keepends=True)))
    n = len(pipe.work_calls())
    r = pipe.run("tst", STUDENT=1, **env)
    assert r.returncode == 2 and "label tables differ" in r.stdout and "mix two table sets" in r.stdout, \
        r.stdout + r.stderr
    assert len(pipe.work_calls()) == n                                          # no stage, no student
    assert not (out / "done_student").exists() and (out / "done_te").exists() == (when == "student")
    assert (out / "logs" / "provenance_diff.log").read_text().strip()
    assert not (pipe.runs / "corr30" / "tst_student" / "tables_provenance.txt").exists()
    r = pipe.run("tst", STUDENT=1, ALLOW_REUSE=1, **env)                        # not a code_rev matter
    assert r.returncode == 2


def test_student_stops_on_tables_rebuilt_during_stage_t(pipe):
    # one launch: a table is rebuilt in place while stage T runs (here: at stage finish) -> the student stage refuses
    meta = pipe.km / "corridor64" / "navtrain_train" / "meta.txt"
    r = pipe.run("tst", STUDENT=1, STUB_ON_FINISH=meta)
    assert r.returncode == 1 and "student: the label tables differ" in r.stdout, r.stdout + r.stderr
    assert "stage student FAILED" in r.stdout and _student_calls(pipe) == []
    out = pipe.out("tst")
    assert (out / "done_finish").is_file() and not (out / "done_student").exists()
    assert "corridor64 navtrain_train" in (out / "logs" / "provenance_diff.log").read_text()
    assert not (pipe.runs / "corr30" / "tst_student" / "tables_provenance.txt").exists()
