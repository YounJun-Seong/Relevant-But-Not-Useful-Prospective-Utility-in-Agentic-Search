# 리뷰 대응 — 코드 실측 확정값 (최종 2026-08-19)

모든 값은 `piu/` 코드와 `results/` 로그에서 직접 산출. 추정치 없음.

## 1. Pairwise utility specialist (= 논문의 frozen conflicting specialist)

정의 위치: `scripts/freeze_models.py:69-77` (`fit()`), 피처 `piu/state_delta.py`,
쌍 구성 `piu/pairwise.py`, 학습쌍 로딩 `scripts/phi_baseline.py:load_phi_pairs`.

**주의: DeBERTa predictor(`piu/predictor.py`)가 아니다.** 그쪽은 "global predictor"
(pred_A, 57–59%). specialist는 φ 위의 절편 없는 로지스틱 회귀.

| 항목 | 실측값 |
|---|---|
| 학습 소스 | `results/gate3` (HotpotQA **train** split, 사전풀 3000 → 선택 1200문항) |
| gate3 전체 쌍 | 66,192 (tie 54,675 / decided **11,517**) |
| decided 쌍의 질문/상태 | 619 questions / 999 states |
| **conflicting specialist 학습쌍** | **5,866 pairs / 524 questions** |
| aligned specialist 학습쌍 | 5,651 pairs / 560 questions |
| 안티시메트릭 증강 후 행 수 | 11,732 (conflicting) / 11,302 (aligned) |
| 모델 | `sklearn.linear_model.LogisticRegression(fit_intercept=False, max_iter=1000)` — L2, C=1.0, lbfgs, tol=1e-4 |
| 실제 수렴 | n_iter 15 (conflicting) / 17 (aligned) |
| 입력 | Δφ = φ(s,d_pref) − φ(s,d_other), 9차원 |
| **유효 차원 7** | Δφ의 `turn_idx`, `n_evidence`는 같은 state 내 쌍이라 **항상 정확히 0** (실측 nonzero=0) → 계수도 0.000 |
| 손실 | binary cross-entropy (샘플 가중치 없음 — margin 가중은 DeBERTa 쪽만 사용) |
| negative 구성 | 각 쌍이 (Δφ→1)과 (−Δφ→0) 두 행으로 들어가는 안티시메트릭 증강 |
| split | train=gate3(train split) / test=gate2(dev split) / 확증=lockbox — 질문 겹침 0 (assert로 강제) |
| seed | 42 (`freeze_models.py` 상단 `torch.manual_seed(42)`, `np.random.seed(42)`) — lbfgs 자체는 결정론적 |
| 동결 시점 | lockbox 생성 **이전**, `results/frozen/specialist_{aligned,conflicting}.joblib` |
| sklearn | 1.9.0 (supplementary Reproducibility와 일치) |

동결 파일 계수 (재확인, results_master C의 "Hotpot dev" 행과 일치):

- conflicting: novelty +0.498, redundancy +1.064, entity_gain −0.244, query_align **−2.537**,
  q0_align −0.208, gap_coverage −0.090, doc_len **−3.366**, turn_idx 0.000, n_evidence 0.000
- aligned: novelty −0.796, redundancy −1.782, entity_gain +0.276, query_align **+4.905**,
  q0_align +2.878, gap_coverage +0.935, doc_len **+5.585**, turn_idx 0.000, n_evidence 0.000

## 2. Continuation horizon + tie handling

정의 위치: `piu/agent.py` (`AgentRunner`, `Episode`), `piu/intervene.py:make_arm_events`,
`scripts/run_gate1.py:make_runner`.

- 개입 후 에이전트는 **새 예산을 받지 않는다.** 검색 예산은 에피소드 전역 `max_turns=4`이고,
  replay Episode는 prefix에서 이미 소비한 검색 수를 그대로 이어받는다
  (`Episode.__init__`의 `search_calls` 집계). 즉 continuation horizon = **4 − (개입 상태까지 쓴 검색 수)**.
- 개입 상태에서 이미 쓴 검색 수 분포 (lockbox, I2_add arm 5,487개): 1회 2,800 / 2회 1,895 / 3회 656 / 4회 136.
- 실제 추가 검색 스텝 수 (lockbox): 0회 2,989 / 1회 1,739 / 2회 608 / 3회 151 — **평균 0.62 스텝**
  (gate2 0.61, gate3 0.65). 즉 대부분의 개입은 0–1회 추가 검색 후 답을 낸다.
- 예산 초과 시: forced-answer 안내 블록을 넣고 즉시 답을 강제 (`AgentRunner.run`).
- 그 외 종료: `max_gen_tokens=768`, 검색당 문서 3개, greedy(temperature 0).

**Tie rule (PIU oracle):** `pi_pick = max(rows, key=U_task_raw_f1)` — Python `max`는 **첫 최대값**을
반환하고, rows 순서는 후보 유형 열거 순서
(bm25_top → dense_top → ce_top → mid_rel → novelty → redundancy → entity_bridge → random)로 고정.
- 동점 빈도: lockbox 686 states 중 argmax 동점 580 (84.5%) — 단 대부분 max=0인 경우.
  **양수 max에서의 동점은 100 states (14.6%)** (gate2 14.9%, gate3 14.6%).
- **중요: Oracle−CE headroom 수치는 tie rule에 영향받지 않는다.** 같은 state에서 base F1이 공통이므로
  U 동점 ⇔ arm_d F1 동점이고, headroom은 F1 값으로 계산된다. 영향받는 것은 "어느 후보 family가 이겼는가"
  라는 귀속(attribution)뿐이다 (아래 5번 경고 참조).
- **CE 쪽은 동점 자체가 없다**: 5개 세팅 3,881 states 전부에서 CE-pick 순위 동점 0건, 미랭크 0건 (실측).

## 3. 발견된 문서-코드 불일치 2건 (수정 필요)

1. **`supplementary.tex:200` — "the preference label is decided by the *controlled* utility difference"
   → 사실과 다름.** `piu/pairwise.py:build_pairs`는 `u["arm_d"]["f1"]`, 즉 **raw** intervention outcome을 쓴다.
   results_master J와도 모순. 리뷰 지적 ③(9% control overlap) 방어의 핵심이 "확증 지표는 raw"인데
   supplement가 반대로 써 있어서, 이 문장은 반드시 "raw intervention outcome"으로 고쳐야 한다.
2. **`supplementary.tex:245` — "logistic models without intercept over *standardized* features"
   → 표준화(z-score) 안 함.** φ는 고정 스케일 정규화(doc_len = log1p(len)/7, turn_idx/4, n_evidence/12,
   entity_gain/20 클리핑)일 뿐이다. "fixed-scale normalized features"로 수정 권장.

## 4. Natural prevalence (리뷰 지적 ④) — stratum별 opportunity rate

state 단위. base EM=1인 state는 base F1=1.0이라 구조상 headroom이 0 (모든 세팅에서 정확히 0.0%).

| 세팅 | 전체 | representative | fail-enriched | base 실패 state만 |
|---|---|---|---|---|
| Hotpot dev (gate2) | 30.1% (174/579) | **18.5%** (51/275) | 40.5% (123/304) | 37.9% (174/459) |
| Hotpot lockbox | 29.3% (201/686) | **22.5%** (75/333) | 35.7% (126/353) | 36.9% (201/544) |
| 2Wiki | 26.4% (216/819) | 28.4% (116/408) | 24.3% (100/411) | 31.8% (216/679) |
| Qwen-14B | 32.1% (269/838) | 25.6% (102/398) | 38.0% (167/440) | 42.0% (269/641) |
| MuSiQue | 21.8% (209/959) | 19.6% (95/484) | 24.0% (114/475) | 23.2% (209/902) |

→ 본문의 18.5%(dev)에 더해 **lockbox representative 22.5%**를 같이 쓸 수 있다. 5세팅 representative
범위는 18.5–28.4%로, stratification이 opportunity를 만들어낸 것이 아님을 보여준다.

## 5. Candidate-family contribution (리뷰 지적 ⑤) — 표 생성 완료

생성: `scripts/family_breakdown.py` → `results/family_breakdown/{report.json,tables.tex}`.
원고 블록 [12]에 반영 완료.

**핵심 방법론 주의**: opportunity state의 14.6%에서 argmax가 동점이다. 파이프라인의 tie rule
(후보 열거 순서 first-max)은 headroom 값에는 영향이 없지만(동점 U ⇔ 동점 F1) **family 귀속은
크게 왜곡한다**. 반드시 tie 균등분할로 보고할 것.

lockbox (686 states / opportunity 201):

| family | mean U | oracle share (tie 분할) | (order-based, 편향) |
|---|---|---|---|
| bm25_top | +0.0180 | 8.9% | 25.4% |
| dense_top | +0.0638 | 25.7% | 33.3% |
| ce_top | +0.0652 | 22.2% | 21.9% |
| mid_rel | +0.0182 | 7.1% | 3.0% |
| novelty | +0.0239 | 8.7% | 6.0% |
| redundancy | +0.0172 | 9.0% | 4.0% |
| entity_bridge | +0.0210 | 9.4% | 4.5% |
| random | +0.0255 | 8.9% | 2.0% |
| **standard 3종** | | **56.8%** | |
| **auxiliary 5종** | | **43.2%** | |

standard vs auxiliary 비중 (5세팅) 및 standard-only headroom (**5세팅 전부 신규 산출**,
기존 defense_analyses는 3세팅만 있었음):

| 세팅 | opp. states | standard | auxiliary | full headroom | standard-only |
|---|---|---|---|---|---|
| HotpotQA dev | 174 | 60.0% | 40.0% | +0.082 [.062,.104] | +0.042 [.029,.058] |
| HotpotQA lockbox | 201 | 56.8% | 43.2% | +0.117 [.092,.142] | +0.078 [.059,.100] |
| 2Wiki | 216 | 50.4% | 49.6% | +0.142 [.117,.168] | +0.083 [.064,.104] |
| Qwen2.5-14B | 269 | 61.8% | 38.2% | +0.097 [.076,.120] | +0.071 [.053,.091] |
| MuSiQue | 209 | 42.0% | 58.0% | +0.097 [.080,.116] | +0.047 [.035,.060] |

- **standard-only가 5세팅 전부 CI 0 제외** → "auxiliary 후보가 결과를 만들었다"는 반박 성립 안 함.
- 단 auxiliary가 무의미한 것도 아니다 (38–58%). MuSiQue가 auxiliary 의존이 가장 크고
  (58%), standard-only 추정치도 full 대비 가장 많이 떨어진다 (+0.097 → +0.047). 이 비대칭을
  숨기지 말고 "conventional pool 안에서도 headroom이 있고, 값싼 state-conditioned probe를 더하면
  더 커진다"로 쓰는 게 정직하다.

## 6. Controlled diagnostic — 소비된 control 민감도 (rollout 불필요)

기존 로그만으로 계산. **질문 단위 추정량**(논문 H1과 동일, paired bootstrap 10K seed 7):

| lockbox | 질문 단위 평균 | 95% CI |
|---|---|---|
| 전체 control | +0.0060 | [−0.0017, +0.0134] |
| 이미 소비된 control(9.1%) 제외 | +0.0051 | [−0.0027, +0.0127] |

arm 단위 단순 평균으로 보면 전체 +0.0104 / 제외 +0.0099 (5세팅 arm 단위 값은 아래 표).
**논문에 쓸 땐 질문 단위 값을 쓸 것** — 다른 모든 수치가 질문 단위 bootstrap이다.

| 세팅 (arm 단위) | 전체 | 미소비만 | 소비된 것 (비율) | ctrl vs ctrl2 부호일치 |
|---|---|---|---|---|
| gate2 | +0.0159 | +0.0153 | +0.0213 (9.2%) | 87.7% |
| lockbox | +0.0104 | +0.0099 | +0.0160 (9.1%) | 88.1% |
| 2Wiki | +0.0038 | +0.0036 | +0.0065 (8.6%) | 89.6% |
| 14B | +0.0204 | +0.0228 | −0.0003 (10.3%) | 87.2% |
| MuSiQue | +0.0022 | +0.0009 | +0.0140 (9.9%) | 91.2% |

## 7. 환경 상태 — 복구 완료 (2026-08-18 18:40)

깨져 있던 `piu/.venv`를 복구했다. **패키지 재설치는 불필요했다** — site-packages(512개)는 온전했고
인터프리터만 사라진 상태였다.

### 복구 절차 (재발 시 그대로 반복)

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv는 ~/.local/bin/uv 에 있음 (PATH에 없음)
uv python install 3.11                  # cpython-3.11-... 별칭 디렉토리까지 재생성 → 기존 심볼릭 링크 자동 복구
cp data/corpus_offsets.npy data/bm25_index/text_offsets.npy   # 리트리버가 찾는 파일, 별도로 유실돼 있었음
hf download Qwen/Qwen2.5-7B-Instruct    # HF 캐시도 함께 유실 (15GB, 약 90초)
```

### 검증 결과

| 항목 | 상태 |
|---|---|
| Python | 3.11.16 (`.venv/bin/python`) |
| torch / vllm | 2.8.0+cu128 / 0.10.2 — CUDA 사용 가능, GPU 2장 인식 |
| transformers / bm25s / sklearn / numpy | 4.55.2 / 0.3.10 / 1.9.0 / 2.2.6 |
| BM25Retriever | `doc_by_title` + `search` 정상 (인덱스 로드 약 1분 45초) |
| 에이전트 루프 | `scripts/smoke_agent.py` → **SMOKE_OK** (base / 주입 replay / replay consistency 전부 통과) |

버전이 supplementary Reproducibility 절 기재값과 **정확히 일치** → 새로 돌리는 실험이 기존 결과와
같은 스택 위에서 나온다.

### 새 실험 돌릴 때 주의

- **머신이 매우 혼잡하다**: load average 68–86, GPU 0은 타 사용자 작업이 68GB 점유 중.
  GPU 1을 쓰고 (`CUDA_VISIBLE_DEVICES=1`), `gpu_memory_utilization`은 0.4 정도로 잡을 것.
- vLLM 엔진 초기화가 부하 때문에 **5–10분** 걸린다. hang으로 오해하기 쉽다.
  torch.compile 캐시는 이제 warm 상태(`~/.cache/vllm/torch_compile_cache/`).
- 로그는 반드시 `python -u`로 받을 것. 버퍼링 때문에 중간에 죽으면 출력이 통째로 사라진다.

---

## 8. 추가 실험 4종 배치 계획 (2026-08-18 확정)

원칙: 본문은 "greedy artifact 아니냐"를 막는 것만, 나머지는 Supplementary.

| 실험 | 상태 | 배치 | 비고 |
|---|---|---|---|
| Decoding stochasticity sensitivity | **실행 완료** (9절) | 본문 Robustness 표 1행 + Supp 상세 | manuscript_edits.tex 블록 [7][8][9] 작성 완료 |
| Matching ablation (rank-bucket/length off) | **실행 완료** (10절) | 본문 1문장 + Supp 표 | manuscript_edits.tex 블록 [10][11] |
| Candidate-family contribution | **표 생성 완료** (5절) | Supp only | manuscript_edits.tex 블록 [12] |
| Stronger prospective gate / rollout proxy | **미실행** | Supp only ("additional gate exploration") | 결과가 메인 스토리를 바꿀 수 있어 본문 배치 위험 |

### 본문 Robustness 표에 추가할 행 (실험 완료 후에만)

```latex
Decoding artifact & Sampled decoding ($T{=}0.7$, $n$ replays) & [TBD] \\
Control matching & Matching spec.\ sensitivity & [TBD] \\
```

### 요약 문장 — 이제 원안 그대로 쓸 수 있다

decoding sensitivity(9절)와 matching ablation(10절)이 둘 다 실행됐으므로:

```latex
Results remain stable under sampled decoding and alternative control-matching
specifications; detailed analyses are provided in the Supplementary Material.
```

남은 미실행 항목은 stronger prospective gate 하나뿐이고, 이건 Supplementary 전용이라 본문 문구에
영향이 없다.

---

## 9. Decoding stochasticity sensitivity — 실행 완료 (2026-08-18)

**프로토콜**: lockbox의 base + I2_add arm 전체(6,173 arms)를 **temperature 0.7, top-p 1.0**로
독립 3회 재실행 (seed 101/102/103). 동결 코드는 수정하지 않고 런타임에 `SamplingParams`만 교체 —
프롬프트, stop 시퀀스, 파싱, 검색, 예산 로직 전부 확증 실행과 동일. control(I3_ctrl)은 재실행하지 않음
(확증 지표는 raw outcome 정의).
스크립트: `scripts/decoding_sensitivity.py` + `scripts/analyze_decoding_sensitivity.py`.
결과: `results/decoding_sensitivity/DECODING_SENSITIVITY_REPORT.json`. 소요 시간 약 13분.

**성격: post-hoc exploratory.** lockbox는 R16에 따라 1회 개봉으로 종료됐고, 이 분석은 새 모델도
새 가설검정도 도입하지 않는 robustness check다. 논문에 그대로 표기할 것.

### Oracle−CE headroom (질문 단위 paired bootstrap 10K, seed 7 — lockbox_eval과 동일 추정량)

| 조건 | Oracle−CE ΔF1 | 95% CI | Opportunity |
|---|---|---|---|
| Greedy (확증값, 재확인) | +0.117 | [+0.093, +0.142] | 29.3% |
| **Sampled, 3회 평균 (같은 rollout에서 선택·평가)** | +0.111 | [+0.091, +0.133] | 39.9% |
| **Sampled, held-out rollout에서 평가 (정직한 수치)** | **+0.033** | **[+0.003, +0.073]** | — |
| Sampled, replicate 1 | +0.127 | [+0.103, +0.154] | 28.3% |
| Sampled, replicate 2 | +0.125 | [+0.101, +0.151] | 27.6% |
| Sampled, replicate 3 | +0.141 | [+0.115, +0.169] | 29.7% |

- ⚠️ **정정(2026-08-20)**: +0.111~+0.141은 *같은 rollout에서 선택하고 평가한* 값이라 노이즈가 큰
  샘플링 조건에서 winner's curse를 포함한다. split-replicate 기준 정직한 수치는 **+0.033**이다.
  greedy에서는 matched-noise null이 +0.002이고 split이 96% 유지되므로 문제가 없다.
  자세한 내용은 `rq_reanalysis_findings.md` 10·14절.
- **Opportunity rate 비교 시 주의**: 3회 평균본의 39.9%는 greedy와 like-for-like가 아니다.
  평균을 내면 F1이 중간값을 갖게 되어 작은 겉보기 이득이 늘어난다 (8후보 × 샘플링 잡음의 max 선택효과).
  공정한 비교는 두 가지다.

| threshold | greedy | sampled (3회 평균) |
|---|---|---|
| >0 | 29.3% | 39.9% |
| ≥0.05 | 29.3% | 39.1% |
| ≥0.10 | 28.7% | 37.2% |
| **≥0.25** | **27.8%** | **27.1%** |

  → ① replicate별 rate 27.6–29.7% vs greedy 29.3%, ② 본문이 이미 쓰는 효과크기 기준 ≥0.25에서
  27.1% vs 27.8%. 둘 다 사실상 동일. **본문에는 ≥0.25 기준을 쓰는 게 가장 정직하고 기존 서술과도 일관.**

### Pairwise 라벨 안정성 (greedy에서 decided된 3,167쌍 대상)

| 지표 | 값 |
|---|---|
| 샘플링에서 정확히 tie로 붕괴 | 21.7% (모순이 아니라 판정 불가) |
| 0이 아닌 margin 중 부호 일치 | **78.0%** |
| τ=0.1에서 여전히 decided | 70.3% |
| 그중 부호 일치 | **80.3%** |

→ **집계 수준 결론(headroom)은 안정적이지만 개별 쌍 라벨은 약 80% 수준**이다. 이건 논문이 이미
주장하는 "realized utility는 state·policy 의존적"과 정확히 일치하므로, 숨기지 말고 그대로 쓰는 게 낫다.
oracle이 8개 후보 중 최선을 고르는 집계 연산이라 개별 쌍의 잡음이 상쇄된다는 설명을 붙일 것.

---

## 10. Matching ablation — 실행 완료 (2026-08-19)

**프로토콜**: lockbox의 5,487개 개입 각각에 대해 matched control을 **완화된 규칙으로 다시 뽑아**
동결 greedy 정책으로 재실행 (16,461 arms, 10분). 개입 arm(+d)은 기존 확증 결과를 그대로 재사용하므로
controlled diagnostic U_ctrl = F1(+d) − F1(+d̃)를 **동일한 개입에 대해** 재계산한다.
스크립트: `scripts/matching_ablation.py` + `scripts/analyze_matching_ablation.py`.
결과: `results/matching_ablation/MATCHING_ABLATION_REPORT.json`.

확증 규칙(full) = BM25(q_t) top-100에서 rank bucket(rank//20) 일치 + 길이 허용오차(10/25/50/100% 단계적)
+ non-gold + title Jaccard ≤ 0.3 + 미사용. 완화 3종:

| spec | 제거한 제약 |
|---|---|
| `no_rank_bucket` | rank bucket |
| `no_length` | 길이 허용오차 |
| `neither` | 둘 다 |

**성격: post-hoc exploratory** (9절과 동일 표기).

### Manipulation check — spec이 실제로 달라졌는지

| spec | rank bucket 일치율 | 길이비 중앙값 | 길이비 ±25% 이내 | fallback |
|---|---|---|---|---|
| no_rank_bucket | 11.0% | 1.00 | 97.0% | 0.3% |
| no_length | 59.4% | 0.83 | 22.3% | 1.8% |
| neither | 9.8% | 0.84 | 20.6% | 0.1% |

의도대로 각 제약만 풀렸다. (no_length의 rank 일치율이 100%가 아닌 이유: 원 코드와 동일하게
후보가 BM25 top-100 밖이면 rank 제약 자체가 적용되지 않는다 — dense_top/novelty/entity_bridge 등.)

### 결과 — controlled diagnostic (질문 단위, n=350)

| matching spec | 평균 | 95% CI |
|---|---|---|
| **full (확증 규칙)** | **+0.0060** | [−0.0017, +0.0137] |
| no_rank_bucket | +0.0099 | [+0.0024, +0.0177] |
| no_length | +0.0089 | [+0.0009, +0.0171] |
| neither | +0.0086 | [+0.0006, +0.0167] |

- **완화해도 결론이 바뀌지 않는다.** 오히려 확증 규칙이 **가장 작은 추정치**를 준다 →
  엄격한 matching은 보수적 방향이었다. 리뷰어가 걱정한 "matching이 결과를 만들었다"의 반대다.
- 정직하게 같이 쓸 것: **full spec의 CI는 0을 포함한다** (+0.0060 [−0.0017, +0.0137]).
  controlled PIU는 애초에 diagnostic이고 확증 지표는 raw outcome이므로 논문 주장에는 영향이 없지만,
  "controlled diagnostic이 유의하다"고 쓰면 안 된다.
