# PARA-SSR 실험 보고서

SSR(ICLR'25)을 PARA-Drive(CVPR'24)식 병렬 aux 구조로 개조하는 작업의 기록.
목적은 **aux task를 대상으로 한 KD 실험용의 깔끔한 병렬 구조**를 갖추는 것이다.

| # | 문서 | 내용 |
|---|---|---|
| 01 | [baseline vs PARA-SSR](01_baseline_vs_para_ssr.md) | 1차 실험 전체 결과, task별 loss·grad norm 분석, 실패 진단 |
| 02 | [Map head를 VAD 스타일로](02_map_head_vad_style.md) | dense가 실패한 이유, VAD 벡터 head 동작 원리, 전환 내역 |
| 04 | [BEV encoder / 격자 구조](04_bev_encoder_and_grid.md) | `bev_embed`의 정의, 좌표 규약, shape 치트시트 — 새 head 작성 시 참조 |
| 05 | [ver2 수정 내역](05_ver2_changes.md) | occ 1채널화, grad scale 증명, map 설계 논의, multibatch 이식, 코드 리뷰 대응 |
| 06 | [Aux 평가 배선](06_aux_evaluation_wiring.md) | map mAP / occ IoU 평가, 학습 중 품질 로깅, shapely 2.x 수정 |
| 07 | [2차 리뷰 검토 및 수정](07_review_round2_fixes.md) | **진단 지표의 의미 오류 수정** — occ 분리도 편향, map 붕괴 감지, gshare 재구성 버그, clipping 계측, 분산 eval 통합 테스트 |

> #03(occupancy teacher 설계)은 미작성. #05 §3.3에 KD 인터페이스 논의가 일부 들어 있다.
>
> **#06의 학습 중 map 지표 해석은 #07에서 정정됐다.** `map_pos_frac`는 폐기됐고
> `map_pts_err_m`는 축별 MAE에서 유클리드로 바뀌었다. 옛 로그를 읽을 때 주의.

## 실험 결과 요약

| 실행 | L2 MAX avg (m) | CR box avg (%) | 비고 |
|---|---:|---:|---|
| SSR 논문 Table 3 (w/o FFP) | 0.75 | 0.20 | 목표 |
| SSR-orig no-FFP, 8 GPU | **0.737** | **0.221** | ✅ 재현 성공 |
| PARA-SSR v1, 8 GPU | 1.000 | 0.514 | ❌ +36% / +133% |
| SSR-orig no-FFP, 2 GPU × b4 | 진행 중 | | multibatch 경로 검증 |
| PARA-SSR ver2 | 미실행 | | occ 1채널 + VAD map |

## 코드 위치

| | 경로 | 브랜치 |
|---|---|---|
| PARA-SSR | `/home/yongjae/e2e/SSR` | `para` |
| 원본 SSR | `/home/yongjae/e2e/SSR-orig` | `main` |
| 참조 구현 B | `/home/yongjae/e2e/SSR-fable` | — |
| work_dirs | `/data1/yong/SSR/work_dirs` (심볼릭 링크, 두 워크트리 공유) | |

## 주의사항

- 실행 시 `NUMBA_CPU_NAME=generic` 필요 (이 CPU에서 numba가 segfault).
- `samples_per_gpu > 1`은 multibatch 수정(`8fe89d5` + para 이식) 이후에만 안전하다.
- GPU 4는 드라이버가 깨져 있다 — `nvidia-smi`에는 보이지만 CUDA 초기화가 실패한다.
- mmcv 함정: `MultiheadAttention(dropout=)` / `ffn_dropout=`이 **클래스 레벨 기본 dict를 변형**한다. 같은 프로세스에서 나중에 만들어지는 레이어가 그 값을 물려받으므로, 두 모델을 한 프로세스에서 비교할 때 반드시 고려할 것.
