# refiner-v3 Step 2 — v3 temporal DET teacher로 CK2 e2e KD 전체 학습

작성: 2026-10-12 00:50 KST (빌더, CPU만 사용; GPU / 학습 launch 없음, git commit 없음).
갱신: 2026-10-12 01:30 KST — 리뷰 반영 (§6: low 2건 수정 = KD object 읽기 fail-loud, smoke temporal gate; nit 2건 유지).
요청 (user, 2026-10-11 밤): "teacher refiner에 대해서 det, map 다 끝나면 알아서 e2e 모델에 KD까지 하는 전체 학습도 알아서 시작해놔".

## 1. 무엇을 바꾸는 실험인가

- **바뀌는 것은 DET teacher 하나뿐**: `ck2T10` → `v3T10_a_learned` (refiner-v3 arm a: causal BEVFusion detection 이력 +
  사전학습 후 frozen learned forecaster + path-time raster). 그 teacher의 KD calibration 파일
  (`ck2/kd_calib/v3T10_a_learned__last.json`, GPU fit 필요)도 같이 바뀐다.
- 나머지는 `ck2e2e_main`과 **완전히 동일**: 같은 `ck_e2e2.yaml`, 같은 recipe(2 GPU, batch 4/GPU, accumulate 16), seed 0,
  ARM=main, BEV KD 없음, MAP teacher `ck2M10` + `ck2M10__last.json` calibration, 30 epoch 스케줄.
- **student는 CK2 그대로** (temporal branch 없음). temporal 정보는 teacher의 KD target(점수 확률 / lateral offset)을 통해서만
  들어간다.
- 결과적으로 ck2e2e_v3T vs ck2e2e_main 비교 = teacher 교체 효과만 분리.

arm set 파일: `tools/ck/e2e2/v3teacher_arm.set` (두 줄: `teacher_det_run`, `kd_calib.det`).

## 2. 코드 변경 (temporal teacher일 때만 동작; non-temporal teacher는 기존 코드와 bit-identical)

| 파일 | 변경 |
|---|---|
| `navsim/agents/para_ssr/ck/e2e_data2.py` | `teacher_temporal(run)` (teacher run config.json의 temporal cfg), `KDObjRows` (object pack reader: `tools/ck/data/obj_pack.ObjPack(split=warmup.packed 이름, source, pack_root, targets=False, expect_tokens=packed token 순서)` → `ObjPack.item(row)`의 obj_in / obj_mask / frame_valid / obj_ok만; **augmentation 없음, fc_* target 파일은 열지도 않음**). `CK2E2ETargetBuilder`는 DET teacher가 temporal일 때만 `kd_obj_in_0 [3,96,17] / kd_obj_mask_0 [3,96] / kd_frame_valid_0 [3] / kd_obj_ok_0 []`를 추가 (row -1 / 범위 밖 → 0 + ok False). MAP teacher가 temporal이면 거부. pack은 builder 생성 시 main process에서 열고 검사(fail fast), worker에는 mmap 없이 pickle. **fail-loud** (리뷰 1): 생성 시 pack fingerprint(build.json sha256 + mmap하는 모든 .npy / tokens.parquet의 size·mtime)를 저장, 각 process(DataLoader worker 포함)의 첫 read 전에 재확인 → pack이 바뀌었거나 없어지면 `RuntimeError`, packed row의 read 오류는 그대로 raise (예전: 조용히 0 + KD off). |
| `navsim/agents/para_ssr/ck/online2.py` | `teacher_forward(net, bev, cand, ego, objs)` = KD teacher ops (trunk.scene → TemporalBranch.scene(objs) → hook → trunk.candidates → score; `CKNet.forward(..., objs=)`와 같은 순서/연산). `TeacherPair2`: load_ck가 PRIVILEGED run을 거부(PermissionError, teacher 이름 포함), MAP temporal 거부, `det_temporal` / `temporal_info`(cfg, branch / forecaster weight sha16, fc_init sha16) 기록, `run(..., objs_det=, obj_ok_det=)` 필수 (DET ok = BEV ok & obj_ok). `kd_objs_of(targets)` = 정확히 OBJ_INPUT_KEYS만 teacher로. `CKE2E2Config.validate`: temporal teacher는 DET만, privileged 거부, TemporalCfg.validate. `CKE2E2.loss`: temporal teacher면 object rows 전달 + 로그 `ck2/kd_obj_ok_frac` (CK2 teacher면 로그도 그대로). fp16 autocast는 기존 `teacher_amp` 그대로 (trainer eval / infer_val_v3도 CUDA fp16 autocast). |
| `tools/ck/e2e2/fit_kd_calib.py` | temporal DET teacher 지원: CKDataset `objects={cfg, aug False, targets False}`로 calibration 토큰(r34 top-16, navtrain_train 학습 rows, 16384 log-stratified)의 clean object rows, forward = `online2.teacher_forward` (TeacherPair2와 같은 함수), obj_ok False 토큰 제외, privileged 거부. inference cache key / JSON에 temporal cfg + pack build.json sha16 + obj 수 기록 (CK2 teacher의 cache key / JSON은 불변). |
| `tools/ck/e2e2/launch_util2.py` | `teacher_check`: temporal teacher는 DET + non-privileged만 (두 모드 모두), `teachers.json det.temporal` = cfg, ckpt 안 temporal branch / forecaster weight sha16, fc_init 파일 sha + **ckpt forecaster == fc_init forecaster 여부**, ca_scales sha, teacher 학습 provenance. `kd_objects_check`: builder와 똑같이 KDObjRows를 열고, pack build.json sha256이 teacher 학습 때 pack(config.json provenance)과 다르면 **거부** → `teachers.json kd_objects`. `smoke-check2`: effective `teacher_det_run`이 temporal이면 gate 3개 추가 (리뷰 2) — `temporal_teacher_recorded` (teachers.json det.temporal: on, non-privileged, branch / forecaster sha16, warnings 없음), `kd_objects_same_as_teacher_training`, `kd_obj_ok_frac_logged_ge_0999` (steps에 `ck2/kd_obj_ok_frac`가 찍혔고 평균 ≥ 0.999). CK2 run은 결과 불변. |
| `tools/ck/e2e2/train_e2e2.sh` | 주석만 (v3 arm 사용법). 동작 변경 없음. |
| `tools/ck/e2e2/smoke_e2e2.sh` | stage `G1M` (main arm만 GPU 0,1; GPU 2,3이 nokd로 사용 중이므로) + `SMOKE_CK_SET_EXTRA=<set 파일>` (g_main.set에 append). 기존 G0–G3 동작 불변. temporal gate는 smoke-check2가 자동 적용 (G1M `pass`에 포함). |
| `tools/ck/e2e2/v3teacher_arm.set` | 신규 (teacher_det_run + kd_calib.det만). |
| tests | `tests/test_v3_e2e_teacher.py` (8), `tests/test_v3_e2e_launch.py` (4), `tests/test_v3_e2e_cpu_dryrun.py` (1), probe `tests/v3e2e_probe.py`. |

수정 금지 파일(model.py, temporal/**, tools/ck/data/**, train_ck2.py, tools/ck/v3/* 등)은 건드리지 않았다.

동작 규칙
- object rows는 teacher가 학습한 것과 같은 train pack (`v3/objpack_k96/bevfusion/navtrain_train`, build sha256
  `d004c23d…`, teacher config.json provenance와 일치 확인)의 **clean** row (KD 시점 augmentation 없음; trainer의 eval /
  infer_val_v3와 같은 입력).
- causal: teacher에는 OBJ_INPUT_KEYS(obj_in, obj_mask, frame_valid)만 들어감. t0 row age 0, 과거 row age > 0을
  TemporalBranch.check_causal이 매 step 강제 (위반 시 예외).
- `kd_obj_ok_0 False` (packed row 없는 토큰 = `strict_rows` 0.999 기준 ≤ 0.1 %, pack의 obj_ok False) → DET ok False →
  그 토큰의 KD 전체 off (기존 "BEV 없으면 KD off" 규칙과 동일). navtrain_train pack의 obj_ok False는 0 / 85,109 [실측].
  pack 변경 / 삭제 / row read 오류는 KD off가 아니라 **예외로 학습 중단** (무인 55 h run이 조용히 KD 없이 도는 것 방지;
  `stop` 후 원인 확인 → `start --resume`).
- forecaster: ckpt 안 weight = fc_init(`v3fc_pre_learned6/forecaster.pt`) weight와 동일 [실측, tensor equal 90/90].

## 3. 테스트 (CPU, `CUDA_VISIBLE_DEVICES=`, 출력은 /tmp만)

```
cd /workspace/yongjae/SSR-v3 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider tests/test_v3_e2e_teacher.py tests/test_v3_e2e_launch.py \
  tests/test_v3_e2e_cpu_dryrun.py --basetemp=/tmp/v3e2e_pytest/all
```

| 테스트 | 내용 | 결과 |
|---|---|---|
| (i) `test_non_temporal_path_bit_identical_to_head` | `git archive HEAD`(/tmp) vs 현재 코드, 별도 프로세스 2개: TeacherPair2 출력(CK2 smoke teacher + **ck2e2e_main 실제 teacher ck2T10/ck2M10 + 실제 calibration**), target builder 배치(2 토큰 + 없는 토큰), warm-up loss / logs / BEV·student gradient 139 tensor | 전부 bit-identical |
| (ii) `test_temporal_teacher_equals_trainer_and_infer_forward` | TeacherPair2(v3) = `train_ck2.forward` (bitwise), = infer_val_v3식 forward(+80 variants) q99.9 ≤ 2e-4, e_lat 일치, MAP 출력은 teacher 교체와 무관 | pass |
| (ii) `test_temporal_teacher_matches_saved_trainer_eval_logits` | CPU fp32 vs teacher가 저장한 GPU fp16 train-eval logits | median ~1.3e-3, max ~0.024 (fp16 noise) [실측] |
| (iii) `test_builder_obj_rows_equal_objpack_item` | kd_obj_* = ObjPack.item = trainer eval ObjectRows, 나머지 key는 CK2 builder와 동일, pickle, 없는 토큰 0 + ok False | pass |
| (iii) `test_kd_obj_rows_fail_loud_on_pack_change_and_read_error` (리뷰 1) | /tmp symlink pack 사본: 실제 pack과 같은 row, 새 process(pickle) read OK → build.json 수정 후 새 process / main 모두 `RuntimeError`, .npy 삭제 → `RuntimeError`, row read 오류 re-raise, row -1은 여전히 0 + ok False | pass |
| (iv) `test_causality_only_object_inputs_reach_the_teacher` | scene에 들어가는 key = OBJ_INPUT_KEYS, fc_*/priv_* 없음, age 규칙, 비인과 row 거부, obj_ok False → 해당 토큰만 KD off | pass |
| (iv) `test_privileged_and_map_temporal_teachers_refused` | oracle(PRIVILEGED) teacher: TeacherPair2 / validate / KDObjRows / fit_kd_calib 모두 거부 | pass |
| (v) `test_e2e_step_with_v3_teacher_set_and_cpu_calibration` | set 파일이 yaml에서 정확히 2 key만 바꿈, fit_kd_calib CPU dry fit(16 토큰, sanity 0.0, cache 재사용), v3 teacher + calibration으로 warm-up e2e step: KD finite, backward finite | pass |
| launch 4개 | teachers.json 기록 (branch / forecaster sha = TeacherPair2가 로드한 것), pack 불일치 / provenance 없음 / privileged / MAP temporal 거부, overrides 차이 = 2 key, **train_e2e2.sh E2E_DRY_RUN이 ck2e2e_main overrides.txt와 2줄만 다름**; 그 dry-run 기록으로 smoke temporal gate (리뷰 2): `kd_obj_ok_frac` 로그 있으면 pass, 없음 / 0.5 / teachers.json 기록 없음 → fail, ck2e2e_main(CK2) → gate 없음 | pass |
| cpu dry run 1개 | 실제 ParaSSRAgent + Lightning(CPU) 2 epoch(warmup, warmup_record) 4 mb: kd_obj_ok_frac 1.0, kd_ok_frac 1.0, KD finite, student state에 temporal 없음 | pass (CPU teacher ~1.4 s/mb) |
| 기존 `tests/test_ck2_e2e_*.py` | 회귀 | 184 passed |

Dry-run diff (`E2E_ROOT=/tmp/v3e2e_dry`, calibration은 /tmp CPU dry fit; run dir / run 이름 정규화) vs
`ck2e2e_main/launch/overrides.txt`:
```
50c50
< ++agent.config.ck_e2e2.teacher_det_run=/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10
> ++agent.config.ck_e2e2.teacher_det_run=/home/external-user/ssd/yongjae_refiner/ck/v3/train/v3T10_a_learned
67c67
< ++agent.config.ck_e2e2.kd_calib.det=/home/external-user/ssd/yongjae_refiner/ck/ck2/kd_calib/ck2T10__last.json
> ++agent.config.ck_e2e2.kd_calib.det=<v3T10_a_learned calibration>
```
(나머지 119줄 동일; `ck_e2e2_effective.json`도 io_dir 외 동일. 실제 launch에서 kd_calib.det는
`/home/external-user/ssd/yongjae_refiner/ck/ck2/kd_calib/v3T10_a_learned__last.json`.)

## 4. arm b 종료 후 main session 실행 순서 (exact)

전제: arm b(`v3T10_b_ca`, GPU 0,1)가 끝났고 (~02:15 KST) Step-1 후처리(infer_val_v3 등)가 GPU 0,1을 쓰고 있지 않을 것.
nokd e2e(GPU 2,3, SSR-ck2)는 그대로 둔다. 모든 명령은 **SSR-v3 worktree**에서 (temporal 모델 코드가 여기에만 있음).

0) 확인
```
nvidia-smi --query-gpu=index,memory.used --format=csv
ls /home/external-user/ssd/yongjae_refiner/ck/v3/train/v3T10_a_learned/done.json
cd /workspace/yongjae/SSR-v3 && RUN=ck2e2e_nokd bash tools/ck/e2e2/train_e2e2.sh status   # 읽기만 (nokd / SSR-ck2 건드리지 않음)
```

1) v3 DET teacher KD calibration fit (GPU 0, ~1–3 분 [추정: ck2T10 46 s inference + temporal 비용])
```
cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /venv/ssr/bin/python tools/ck/e2e2/fit_kd_calib.py \
  --run /home/external-user/ssd/yongjae_refiner/ck/v3/train/v3T10_a_learned --which last --n-tokens 16384 \
  2>&1 | tee /home/external-user/ssd/yongjae_refiner/ck/ck2/kd_calib/v3T10_a_learned__last.log
```
→ `/home/external-user/ssd/yongjae_refiner/ck/ck2/kd_calib/v3T10_a_learned__last.json` (= set 파일 경로).
확인: 로그 `sanity {'fwd_score_maxdiff_vs_cknet': 0.0}`, `data.n_tokens_obj_ok == data.n_tokens`, nc / ep / ttc
held-out NLL after < before (ck2T10 기준: nc .0554→.0475, ep .426→.410, ttc .158→.134).

2) launcher dry run + main과 diff (아무것도 launch 안 함; `$R/launch_dryrun/`만 씀, 이후 실제 start를 막지 않음)
```
cd /workspace/yongjae/SSR-v3 && RUN=ck2e2e_v3T E2E_DRY_RUN=1 CK2_E2E_APPROVED=1 \
  E2E_CK_SET=/workspace/yongjae/SSR-v3/tools/ck/e2e2/v3teacher_arm.set bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1
E=/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e
diff <(sed "s#$E/ck2e2e_main#<R>#g; s#ck2e2e_main#<RUN>#g" $E/ck2e2e_main/launch/overrides.txt) \
     <(sed "s#$E/ck2e2e_v3T#<R>#g; s#ck2e2e_v3T#<RUN>#g" $E/ck2e2e_v3T/launch_dryrun/overrides.txt)
```
기대: teacher_det_run, kd_calib.det 2줄만 (W&B API가 안 되면 `wandb.mode=offline` 1줄이 더 나올 수 있음 — main은 online).
`$E/ck2e2e_v3T/launch_dryrun/teachers.json`에서 `det.temporal.forecaster_equals_fc_init: true`, `warnings: []`,
`kd_objects.same_as_teacher_training: true` 확인.

3) GPU smoke (E2E_SMOKE=1, GPU 0,1, main arm 3 epoch × 300 mb/rank, labeler 포함; ~20–30 분 [추정])
```
cd /workspace/yongjae/SSR-v3 && TAG=v3T_$(TZ=Asia/Seoul date +%m%d_%H%M) \
  SMOKE_CK_SET_EXTRA=/workspace/yongjae/SSR-v3/tools/ck/e2e2/v3teacher_arm.set \
  bash tools/ck/e2e2/smoke_e2e2.sh G1M; echo rc=$?
```
→ `/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke/<TAG>/result_G1M_main.json` `pass: true` **AND rc=0이어야 4단계로**
(train exit 0, non-finite 0, skipped 0, 3 phase, mem ≤ 22 GB, on-policy label prev ≥ 90 %, gen round trip bit-exact, 그리고
temporal gate `temporal_teacher_recorded` / `kd_objects_same_as_teacher_training` / `kd_obj_ok_frac_logged_ge_0999`가
모두 true — `checks`에 이 3개 key가 **없으면** temporal teacher가 안 잡힌 것이므로 중단).
```
python3 -c "import json;d=json.load(open('/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke/<TAG>/result_G1M_main.json'));print(d['pass'],{k:v for k,v in d['checks'].items()},d.get('temporal'))"
```
추가 확인 (참고):
```
S=/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke/<TAG>/g1m_main
grep -c '"ck2/kd_obj_ok_frac": 1.0' $S/ck_e2e2/steps_rank0.jsonl; wc -l $S/ck_e2e2/steps_rank0.jsonl
grep "temporal DET teacher" $S/launch/train.log | head -2
python3 -c "import json;d=json.load(open('$S/../result_G1M_main.json'));print(d['phase_timing'],d['estimate_30ep_2gpu'])"
```
비교 기준 (ck2e2e_main on-policy 실측): teacher_ms median 58.6, sec_step median 0.556 s, peak mem 14.2 GB, kd_ok_frac 1.0.
smoke가 끝나면 GPU 0,1이 비었는지 다시 `nvidia-smi`.

4) 본 학습 launch (GPU 0,1)
```
cd /workspace/yongjae/SSR-v3 && tar czf /home/external-user/ssd/yongjae_refiner/ck/ck2/e2e/code_snapshot_e2e_v3T_$(TZ=Asia/Seoul date +%Y%m%d_%H%M).tar.gz \
  navsim tools tests docs/refiner_v3      # main run과 같은 방식의 코드 스냅샷 (worktree에 미커밋 변경이 있으므로)
cd /workspace/yongjae/SSR-v3 && RUN=ck2e2e_v3T CK2_E2E_APPROVED=1 \
  E2E_CK_SET=/workspace/yongjae/SSR-v3/tools/ck/e2e2/v3teacher_arm.set bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1
RUN=ck2e2e_v3T bash tools/ck/e2e2/train_e2e2.sh status
```
launch 후 확인: `$E/ck2e2e_v3T/launch/teachers.json` (det.temporal, kd_objects), train.log의
`[ck_e2e2] temporal DET teacher ...` / `KD calibration: det .../v3T10_a_learned__last.json` 줄,
`ck_e2e2/steps_rank0.jsonl`의 `ck2/kd_obj_ok_frac`(~1.0), `ck2/kd_ok_frac`, `ck2/teacher_ms`.
예상 시간: ck2e2e_main 52.6 h (10-09 03:34 → 10-11 08:12 KST) × (1 + temporal teacher 오버헤드) [추정 55–60 h; smoke
`estimate_30ep_2gpu`로 갱신].
중단 / 재개: `RUN=ck2e2e_v3T bash tools/ck/e2e2/train_e2e2.sh stop` / `... start --resume --gpus 0,1` (같은 E2E_CK_SET 필요).

## 5. 남은 리스크 / 주의

- **pack 재빌드 금지**: 학습 중 `v3/objpack_k96/bevfusion/navtrain_train`을 지우거나 다시 쓰면 (build_obj_pack.py는
  complete pack 덮어쓰기를 거부하므로 의도적 삭제가 필요) 다음 worker / epoch의 첫 read에서 `RuntimeError`로 학습이
  멈춘다 (조용히 틀린 row / KD off로 계속 도는 일은 없음). 이미 열린 worker는 열어둔 mmap(옛 inode)을 계속 읽는다.
  launcher는 start / resume 때 build sha256을 teacher 학습 pack과 다시 비교한다. build_obj_pack.py 수정 중인 다른 workflow 주의.
- **코드 고정**: 학습은 SSR-v3 worktree 코드를 그대로 import한다 (labeler 재시작, resume도 현재 파일을 읽음). 학습 중
  `online2.py`, `e2e_data2.py`, `model.py`, `temporal/**`, `tools/ck/data/obj_pack.py`를 바꾸지 말 것 (4단계 스냅샷 권장).
- fp16: teacher는 CUDA fp16 autocast (teacher 학습 / eval / infer_val_v3와 동일). CPU fp32와는 logit max ~0.024 차이 [실측].
- 속도 / 메모리: temporal teacher forward(object forecast + raster + st_enc)가 매 step 추가. smoke에서 mem ≤ 22 GB,
  teacher_ms 확인. calibration 파일이 없으면 launcher가 거부한다 (1단계 필수).
- calibration도 teacher 교체의 일부: DET 쪽 Platt 계수가 v3 teacher 기준으로 새로 fit됨 (MAP은 그대로).
- packed row가 없는 토큰(≤ 0.1 %)은 KD off (main은 BEV가 있으면 KD on이었음) — 무시 가능한 차이.
- 미커밋: 모든 변경은 worktree에만 있음 (commit 안 함).

## 6. 리뷰 반영 (2026-10-12 01:30 KST)

| # | 심각도 | 내용 | 조치 |
|---|---|---|---|
| 1 | low | `KDObjRows.item`이 모든 예외를 삼켜 0 + `kd_obj_ok_0 False` 반환 → pack이 바뀌거나 없어지면 ~55 h 동안 DET KD가 조용히 꺼질 수 있음 (`ck2/kd_obj_ok_frac`는 로그만) | **수정**: per-process fingerprint 재확인 (build.json sha256 + 파일 size / mtime) → 불일치 / 삭제 시 `RuntimeError`; row read 오류 re-raise; 0 반환은 row -1 / 범위 밖만. 테스트 1개 추가. non-temporal 경로 무관 (KDObjRows는 temporal teacher에서만 생성) |
| 2 | low | G1M smoke `pass: true`가 temporal KD 경로 실행을 증명하지 않음 | **수정**: `smoke-check2`에 temporal gate 3개 (effective teacher_det_run이 temporal일 때만, CK2 run 결과 불변). §4-3에서 gate 필수 |
| 3 | nit | W&B tags가 arm에서 나와 v3T run도 `[ck2e2e, main]` (`wandb.name`만 다름) | 유지 (overrides 2줄 diff 보존). 학습 후 W&B에서 `v3T` tag 수동 추가 |
| 4 | nit | 테스트 (ii) infer 경로 허용오차 2e-4 / 1e-3 (CPU 실제 차이 0) | 유지. GPU fp16 일치는 1단계 `fwd_score_maxdiff_vs_cknet == 0.0`을 **필수 조건**으로 확인 |

재실행 (01:30–01:50 KST, CPU, /tmp만): 신규 `tests/test_v3_e2e_*.py` **13 passed** (218 s), 기존 `tests/test_ck2_e2e_*.py`
**184 passed** (882 s). train_e2e2.sh `E2E_DRY_RUN=1` (E2E_ROOT=/tmp, kd_calib.det = /tmp CPU dry fit): rc 0, 121줄,
ck2e2e_main 대비 diff = 50행 `teacher_det_run`, 67행 `kd_calib.det` 2줄뿐; `ck_e2e2_effective.json` 차이 =
teacher_det_run / kd_calib / io_dir / _ck_sets; teachers.json `forecaster_equals_fc_init: true`, `warnings: []`,
`kd_objects.same_as_teacher_training: true` (pack `d004c23d…`). 실제 kd_calib / e2e 디렉터리에는 아무것도 쓰지 않음.

