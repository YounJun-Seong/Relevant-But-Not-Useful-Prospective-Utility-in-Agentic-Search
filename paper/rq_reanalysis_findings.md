# RQ2/RQ3 재분석 — 실측 결과 (2026-08-20)

리뷰어 W1/W2/W4 검증. 모든 수치는 코드/로그 실측. 신규 rollout은 W4(state-aware ranker)뿐이고
나머지는 기존 개입 결과 재사용.

신규 스크립트: `sanity_audit.py`, `ce_scores_candidates.py`, `build_phi_candidates.py`,
`rq2_reanalysis.py`, `rq2_ce_adjusted.py`, `rq3_state_gate.py`, `state_aware_relevance.py`,
`analyze_state_aware.py`. 결과: `results/{sanity_audit,rq2_reanalysis,rq2_ce_adjusted,rq3_state_gate,state_aware_relevance}/`.

---

## 1. W1 확정 — pair-level conflict 정의는 순환적이다

`phi_baseline.load_phi_pairs`가 `ce_agrees = (CE가 PIU-선호 문서를 더 높게 평가)`로 라벨을 만들고,
쌍은 항상 "첫 번째가 선호됨" 방향으로 정렬된다. 따라서:

| 세팅 | follow-CE (aligned) | invert-CE (conflicting) | conflicting specialist |
|---|---|---|---|
| dev | **1.000** | 0.799 | 0.665 |
| lockbox | **1.000** | 0.813 | 0.692 |
| 2Wiki | **1.000** | 0.822 | 0.660 |
| 14B | **1.000** | 0.817 | 0.667 |
| MuSiQue | **1.000** | 0.805 | 0.666 |

- aligned subset에서 "CE 따르기"는 **정확히 100%** — 정의상 자명.
- conflicting subset에서 "CE 뒤집기"는 80~82%. 100%가 아닌 이유는 아래 2번(라벨 오염) 때문이며,
  **오염을 제거하면 100%가 된다.**
- **specialist(66~69%)는 5개 세팅 전부에서 invert-CE(80~82%)보다 나쁘다.**

## 2. 추가 발견 — CE 라벨 오염 (top-20 절단)

`ce_rankings.jsonl`은 ~109개 pool 중 **CE top-20만** 저장한다. 후보 8개 중 top-20 밖 문서는
`order.get(doc, 10**6)` 센티넬을 받으므로, **두 문서가 모두 top-20 밖이면 `rp < ro`가 False가 되어
"conflicting"으로 분류된다.** CE가 의견이 없는 경우가 불일치로 둔갑한다.

| 세팅 | 두 문서 모두 top-20 밖 | conflicting 중 해당 비율 |
|---|---|---|
| dev | 474 | **40.1%** |
| lockbox | 613 | **37.4%** |
| 2Wiki | 600 | 35.6% |
| 14B | 708 | 36.7% |
| MuSiQue | 787 | 39.1% |

→ 보고된 "CE-conflicting pair"의 **35~40%는 진짜 불일치가 아니다.**

**해결**: 동일 cross-encoder로 후보 8개의 CE 점수를 재계산 (`ce_scores_candidates.py`).
저장된 top-20 순위를 25,692쌍 중 25,690쌍에서 재현 (fp16 동점 2건 차이) → 신뢰 가능.
이후 모든 분석은 이 정확한 점수를 쓴다. 정확한 점수 기준 conflicting 비율은 37.4~47.0%로
기존(46.9~58.0%)보다 낮다.

## 3. 추가 발견 — Opportunity 정의가 캡션과 코드가 다르다

`results_section_v2.tex` 캡션: "fraction of states with at least one candidate that yields a better
downstream outcome than **the CE-selected one**". 그러나 코드(`lockbox_eval.py`)는
`max(U_task_raw_f1) > 0`, 즉 **base 대비** 개선을 센다.

| 세팅 | >base (공표값) | >CE-selected (캡션 서술) |
|---|---|---|
| dev | 30.1% | 14.0% |
| lockbox | 29.3% | 18.8% |
| 2Wiki | 26.4% | 20.0% |
| 14B | 32.1% | 16.7% |
| MuSiQue | 21.8% | 19.5% |

→ 캡션을 코드에 맞추거나(권장: "than the pre-intervention trajectory"), 숫자를 캡션에 맞춰
교체해야 한다. 같은 리뷰어가 다음에 찌를 지점.

## 4. W2 해소 — 76.2 / 56.7 / 56.3 재구성

`lockbox_eval.py` H3 기준:
- `ceiling`(76.2%) = (n_aligned × aligned-specialist acc + n_conflicting × conflicting-specialist acc)/n
  = (1528 × 0.838 + 1639 × 0.692)/3167. **산술 모순 없음** — 리뷰어의 산수 의심은 해소 가능.
- `baseline`(56.7%) = aligned specialist를 전체 쌍에 적용.
- `best_mixture`(56.3%) = gate 가중 혼합.

**그러나 더 큰 문제**: 같은 oracle 조건에서 trivial 규칙(aligned→follow CE, conflicting→invert CE)은

| 세팅 | specialist ceiling | **trivial ceiling** |
|---|---|---|
| dev | 0.760 | **0.906** |
| lockbox | 0.762 | **0.903** |
| 2Wiki | 0.712 | **0.907** |
| 14B | 0.752 | **0.912** |
| MuSiQue | 0.717 | **0.887** |

→ 보고된 "oracle-condition ceiling"은 자명한 규칙보다 14~20pp 낮다. 현재 형태로는 유지 불가.

---

## 5. RQ2 재정의 결과 — unconditional specialist는 CE를 못 이긴다

conflict 조건 없이 **모든 decided pair**(gate3 11,517쌍)로 φ 로지스틱 재학습.
선택은 tournament와 동치인 `argmax_d w·φ(d)`.

### (a) pairwise accuracy (전체 decided pairs, 정확한 CE 점수)

| 세팅 | uncond. specialist | follow-CE | invert-CE | conflict-conditioned |
|---|---|---|---|---|
| dev | 0.609 | **0.626** | 0.374 | 0.410 |
| lockbox | 0.561 | **0.575** | 0.425 | 0.441 |
| 2Wiki | 0.531 | **0.567** | 0.433 | 0.453 |
| 14B | 0.603 | **0.608** | 0.392 | 0.410 |
| MuSiQue | 0.524 | **0.530** | 0.470 | 0.473 |

**5개 세팅 전부 CE 단독이 specialist와 같거나 낫다.** 기존 conflict-conditioned specialist를
무조건 과제에 적용하면 chance 이하(0.41~0.47) — CE를 뒤집는 법을 배웠다는 증거.

### (b) 선택기별 realized F1 (질문 단위)

| lockbox 선택기 | abs F1 | CE 대비 gain |
|---|---|---|
| anti-oracle | 0.2964 | −0.1430 [−0.169,−0.117] |
| base (개입 없음) | 0.3758 | −0.0636 [−0.093,−0.035] |
| lowest-CE | 0.3924 | −0.0470 [−0.072,−0.021] |
| random | 0.3993 | −0.0401 [−0.063,−0.017] |
| **CE** | **0.4394** | 0 |
| specialist (uncond.) | 0.4342 | −0.0052 [−0.025,+0.014] |
| **oracle** | **0.5560** | **+0.1166** |

- **CE는 random(0.399), lowest-CE(0.392), base(0.376)를 확실히 이긴다** → W3("CE가 사실상 random 아니냐")
  에 대한 직접적 답. 5개 세팅 모두 동일 패턴.
- specialist는 CE를 이기지 못한다 (5세팅 gain: −0.005 ~ −0.022, 3개는 유의하게 음수).

### (c) top-1 선택 정확도 (any-max)

| 세팅 | random | lowest-CE | CE | specialist |
|---|---|---|---|---|
| lockbox | 0.755 | 0.742 | **0.812** | 0.800 |

(5세팅 전부 CE ≥ specialist. 동점 max가 많아 절대값은 높게 나옴.)

## 6. Anti-alignment / brevity 시그니처는 conditioning 산물이다

### (a) CE 점수와 feature의 후보 수준 상관 (Spearman)

| 세팅 | query_align | doc_len | q0_align |
|---|---|---|---|
| dev | **+0.743** | +0.368 | +0.562 |
| lockbox | **+0.725** | +0.332 | +0.533 |
| 2Wiki | **+0.659** | +0.272 | +0.448 |
| 14B | **+0.751** | +0.361 | +0.528 |
| MuSiQue | **+0.735** | +0.316 | +0.407 |

CE 점수는 query alignment와 강하게 양의 상관 → "CE와 불일치"로 조건화하면 anti-alignment가
**기계적으로 유도된다.** 리뷰어 지적이 데이터로 뒷받침된다.

### (b) 무조건 분석에서의 계수 (괄호 = bootstrap 부호 안정성)

| 세팅 | query_align (φ만) | query_align (CE 통제) | doc_len (φ만) | doc_len (CE 통제) |
|---|---|---|---|---|
| dev | −0.00 (0.56) | −0.22 (0.99) | **+0.14** (0.98) | +0.06 (0.77) |
| lockbox | **+0.14** (1.00) | −0.01 (0.55) | **+0.22** (1.00) | +0.15 (0.99) |
| 2Wiki | **+0.19** (1.00) | −0.01 (0.57) | −0.04 (0.84) | −0.14 (1.00) |
| 14B | +0.00 (0.50) | −0.23 (1.00) | **+0.20** (1.00) | +0.11 (0.98) |
| MuSiQue | **+0.17** (1.00) | +0.06 (0.85) | **+0.18** (1.00) | +0.15 (1.00) |

- **query_align이 음수가 아니다.** conflicting subset 안에서는 5세팅 모두 −1.9~−2.9였으나,
  무조건 분석에서는 0 또는 양수. CE 통제 후에도 5세팅 중 2개만 안정적 음수.
- **doc_len은 방향이 뒤집힌다.** "brevity" 주장과 반대로 4/5 세팅에서 양수(긴 문서가 유리).

→ 현재의 "anti-alignment and brevity are robust signatures" 주장은 **무조건 분석에서 재현되지 않는다.**
conflicting 조건 *안에서* 성립한다는 서술은 사실이지만, 그 조건이 CE와 강하게 상관된 outcome-정의
조건이므로 utility에 대한 진술로 읽히면 안 된다.

### (c) CE + φ 결합 선택기도 CE를 못 이긴다

| 세팅 | CE+φ gain | φ only gain |
|---|---|---|
| dev | −0.0186 [−0.039,+0.000] | −0.0220 |
| lockbox | −0.0011 [−0.020,+0.017] | −0.0042 |
| 2Wiki | −0.0105 [−0.029,+0.008] | −0.0159 |
| 14B | −0.0161 [−0.035,+0.001] | −0.0194 |
| MuSiQue | −0.0064 [−0.017,+0.004] | −0.0131 |

---

## 7. RQ3 재정의 — state-level relevance failure

`Z(s_t) = 1[ H(s_t) > 0 ]`, `H = Y(d_PIU) − Y(d_CE)`. 순환성 없음(CE 순위 뒤집기로 풀리지 않음).
Gate는 개입 전 관측치만 사용: CE 통계(max/mean/std/margin), φ 후보집합 aggregate(mean/max/min/std),
후보 수, probe(가능한 세팅만). gate3 states로 학습(2,368 states, Z=17.8%).

| 세팅 | states | Z rate | gate AUC | AUC(Z≥0.25) | prec@prevalence | probe |
|---|---|---|---|---|---|---|
| dev | 579 | 0.140 | 0.599 | 0.599 | 0.185 | 있음 |
| lockbox | 686 | 0.188 | **0.637** | 0.633 | 0.287 | 있음 |
| 2Wiki | 819 | 0.200 | 0.527 | 0.532 | 0.256 | 없음 |
| 14B | 838 | 0.167 | 0.560 | 0.556 | 0.179 | 없음 |
| MuSiQue | 959 | 0.195 | **0.505** | 0.510 | 0.187 | 없음 |

→ 비순환 라벨에서도 prospective identification은 약하다 (AUC 0.51~0.64). 기존 결론과 방향 일치.

### 라우팅 (lockbox, 질문 단위 realized F1)

| 라우터 | abs F1 | CE 대비 gain |
|---|---|---|
| CE only | 0.4394 | 0 |
| always specialist | 0.4342 | −0.0052 |
| gate → specialist | 0.4375 | −0.0019 [−0.011,+0.007] |
| oracle state → specialist | 0.4719 | **+0.0324 [+0.020,+0.046]** |
| oracle | 0.5560 | +0.1166 |

### ⚠️ 플라시보 검정 — oracle-state 이득은 선택 효과다

`Z`가 `max_d Y(d) > Y(d_CE)`로 정의되므로, Z=1인 state에서는 **아무 후보나** 골라도 CE보다 나아 보인다.
true Z에서 랜덤/최저CE 후보를 라우팅한 플라시보:

| 세팅 | oracle-state (specialist) | 플라시보: random | 플라시보: lowest-CE |
|---|---|---|---|
| dev | +0.0183 | +0.0213 | +0.0183 |
| lockbox | +0.0324 | **+0.0321** | +0.0389 |
| 2Wiki | +0.0327 | **+0.0307** | +0.0465 |
| 14B | +0.0265 | **+0.0241** | +0.0281 |
| MuSiQue | +0.0210 | **+0.0248** | +0.0297 |

→ **specialist는 랜덤과 구분되지 않는다.** "완벽한 state 식별이 있으면 이득이 있다"고 주장하면 안 되며,
oracle-state 수치는 반드시 플라시보와 함께 보고해야 한다.

---

## 8. W4 — state-aware relevance baseline (신규 실험, lockbox)

Qwen2.5-7B에 q0 + 현재 질의 + 이미 소비한 근거를 주고 후보 8개를 **relevance 기준** listwise 랭킹
(프롬프트에서 downstream 정확도 예측/미래 검색 시뮬레이션 금지 명시). greedy, state당 1콜, 686 states 17초.

| | abs F1 | Oracle 대비 headroom |
|---|---|---|
| CE (query-local, 확증 baseline) | 0.4394 | +0.1166 [+0.092,+0.142] |
| **State-aware LLM ranker** | **0.4228** | **+0.1333 [+0.108,+0.160]** |
| Oracle | 0.5560 | — |

- CE와 top-1 일치율 45.9% → 실제로 다른 랭커이지 CE 복제가 아니다.
- **headroom이 닫히지 않는다. 오히려 커진다.** state를 다 줘도 회수되지 않으므로,
  "state-blind relevance만의 문제"라는 반론은 성립하지 않는다.
- 한계: 랭커가 에이전트와 같은 모델 계열(Qwen2.5-7B) 1종. 더 강한 reranker는 미검증.

**미완료**: dev/2Wiki/14B/MuSiQue의 state-aware 실행은 GPU 점유(타 사용자 작업이 양 GPU 65GB씩)로
대기 중. lockbox가 확증 세팅이라 핵심 수치는 확보됨.

---

## 9. 종합 판단

- **RQ1은 그대로 유지된다.** Oracle−CE +0.082~+0.142, 그리고 CE ≫ random ≫ base라는 절대값 사다리가
  새로 확보되어 오히려 더 강해졌다.
- **RQ2는 현재 형태로 유지 불가.** conflict-conditioned 정의는 순환적이고, 무조건 재정의 시
  specialist는 CE를 이기지 못한다. 시그니처 2종도 재현되지 않는다.
- **RQ3는 비순환 형태로 재구성 가능하고 결론이 유지된다** (AUC 0.51~0.64, routing 회수 ≈0).
  단 oracle-state 이득은 플라시보와 구분되지 않으므로 "식별만 되면 회수 가능"이라고 쓸 수 없다.
- 따라서 성립하는 이야기는: **headroom은 실재하고 크지만, 테스트한 값싼 근사는 ranking에서도
  identification에서도 아무것도 회수하지 못한다.** RQ2를 positive claim으로 살리는 것보다 이 편이
  데이터에 부합하고, W1/W2를 동시에 제거한다.

---

# 리뷰어 최우선 5항목 — 실행 결과 (2026-08-20 후반)

LaTeX 표: `results/review_tables/tables.tex` (4종 자동 생성).

## 10. Placebo oracle — "max-of-noise 아니냐"에 대한 직접 방어 ✅

같은 137개 lockbox audited states에서 greedy 3회 replay와 sampled 3회 replicate를 비교.

| 정책 | select+eval 동일 | **split** (한 rollout에서 선택, 나머지로 평가) | **matched-noise NULL** | 후보내 SD |
|---|---|---|---|---|
| **Greedy (T=0, 논문 정책)** | +0.136 | **+0.130 (96.1% 유지)** | **+0.0024** | 0.0013 |
| Sampled (T=0.7) | +0.112 | +0.033 (29.2%) | +0.0725 | 0.0648 |

**핵심**: greedy에서 matched-noise null은 **+0.002**, 즉 8개 후보에 대한 max 연산이 노이즈만으로
만들어내는 이득은 사실상 0이다. greedy 정책에서 결과가 거의 결정론적이기 때문(후보내 SD 0.0013).
split 검정도 96% 유지. **"Oracle gain이 max-of-noise"라는 공격은 논문의 실제 정책에서 성립하지 않는다.**

sampled 대비는 이 검정에 **검정력이 있음을 보여주는 양성 대조**다 — 노이즈가 실제로 있으면
(SD 0.065) null이 +0.073으로 치솟고 split이 71% 붕괴한다. 같은 검정이 greedy에서 0을 준다는 점이
결과의 신뢰도를 높인다.

전체 세팅 permutation null (1,000회, greedy):

| 세팅 | observed | N1 identity-permutation null | N2 iid-redraw null |
|---|---|---|---|
| dev | +0.082 | +0.154 | +0.119 |
| lockbox | +0.117 | +0.154 | +0.120 |
| 2Wiki | +0.142 | +0.166 | +0.129 |
| 14B | +0.097 | +0.154 | +0.121 |
| MuSiQue | +0.097 | +0.120 | +0.092 |

N1/N2는 후보-결과 연결을 끊으므로 CE 선택도 무작위가 된다 → 이 null들이 observed보다 **크다**는 것은
**CE가 무작위보다 확실히 낫다**는 뜻(W3 방어). 단 이들은 max-of-noise 검정이 아니므로 위의
matched-noise/split과 구분해서 보고할 것.

## 11. Null-shuffle simulation — RQ2 시그니처 ❌ (기각)

CE 점수와 φ는 그대로 두고 **state 내에서 결과만 permute** → decided pair와 CE-conflicting subset을
파이프라인과 동일하게 재구성 → 계수 재적합. 300회.

| 세팅 | query_align 실제 | query_align **null 95% CI** | doc_len 실제 | doc_len **null 95% CI** |
|---|---|---|---|---|
| dev | −4.93 | [−5.70, −3.49] | −3.30 | [−5.30, −2.61] |
| lockbox | −5.49 | [−6.00, −4.15] | −4.36 | [−5.93, −3.45] |
| 2Wiki | −3.32 | [−5.21, −3.54] | −5.13 | [−5.60, −3.07] |
| 14B | −5.25 | [−5.75, −4.10] | −3.97 | [−5.93, −3.47] |
| MuSiQue | −4.80 | [−5.99, −4.19] | −3.67 | [−5.67, −3.13] |

**실제 계수가 null 분포 안에 있다.** null이 실제만큼 극단적일 확률은 query_align 14.7~100%,
doc_len 17.3~83.0% — 어느 세팅도 유의하지 않다.

→ **utility와 φ 사이에 아무 관계가 없어도**, CE-conflicting으로 조건화하는 것만으로
query_align ≈ −4.5, doc_len ≈ −4.4가 나온다. 두 "confirmatory signature"는 조건화 산물이며
**논문에서 findings로 유지할 수 없다.**

부수: exploratory로 분류했던 redundancy/novelty가 오히려 일부 세팅에서 null을 벗어난다
(redundancy 0.3~1.3%, novelty 1.0~1.7%). 그러나 부호가 세팅마다 달라 안정적 주장 불가.

## 12. 강한 / state-aware relevance baseline ✅ (5 settings 완성)

BAAI/bge-reranker-v2-m3(568M)를 두 가지 질의 형태로, 그리고 Qwen2.5-7B listwise를 5개 세팅 전부.

Oracle 대비 headroom:

| 세팅 | random | **CE (MiniLM, q_t)** | **BGE (q_t)** | **BGE (state-aware)** | **Qwen listwise (state)** |
|---|---|---|---|---|---|
| dev | +0.153 | +0.082 | +0.099 | +0.087 | +0.095 |
| lockbox | +0.157 | +0.117 | +0.118 | +0.116 | +0.133 |
| 2Wiki | +0.172 | +0.142 | +0.135 | +0.136 | +0.141 |
| 14B | +0.161 | +0.097 | +0.096 | **+0.090** | +0.100 |
| MuSiQue | +0.121 | +0.097 | +0.099 | +0.104 | +0.105 |

- **강한 reranker도, state를 다 준 reranker도 headroom을 닫지 못한다.** 최대 개선은 14B에서
  −0.008(+0.097→+0.090)이고, 절반 이상의 칸에서는 오히려 CE보다 headroom이 크다.
- 절대 F1에서 CE ≫ random이 5세팅 전부 성립 → CE는 진지한 baseline이다.
- 따라서 claim을 "query-local relevance의 한계"로 좁힐 필요가 **없다.** 강한/state-aware까지
  포함해 "tested relevance selectors" 전체로 말할 수 있다.

## 13. Representative sample headroom ✅

| 세팅 | pooled | **representative** | failure-enriched |
|---|---|---|---|
| dev | +0.082 [.062,.104] | +0.062 [.035,.092] | +0.103 [.073,.134] |
| **lockbox** | +0.117 [.092,.142] | **+0.119 [.085,.156]** | +0.115 [.082,.150] |
| 2Wiki | +0.142 [.117,.168] | **+0.159 [.123,.197]** | +0.126 [.092,.162] |
| 14B | +0.097 [.076,.120] | +0.087 [.058,.119] | +0.108 [.077,.142] |
| MuSiQue | +0.097 [.080,.116] | +0.096 [.073,.121] | +0.098 [.071,.128] |

- **representative stratum만으로도 5세팅 전부 CI가 0을 제외**하고, 크기도 pooled와 사실상 같다
  (lockbox +0.119 vs +0.117; 2Wiki에서는 representative가 더 큼).
- dev만 enrichment 효과가 보인다(+0.062 vs +0.103). 확증 세팅인 lockbox에서는 차이가 없다.
- → failure-enriched sampling이 결과를 만들었다는 우려는 데이터로 기각된다.

## 14. ⚠️ 앞서 드린 decoding sensitivity 수치 정정

이전에 "sampled decoding에서 Oracle−CE +0.111로 불변"이라고 보고했으나, 그 값은
**같은 rollout에서 선택하고 평가한 값**이라 노이즈가 큰 조건에서 winner's curse를 포함한다.
정직한 수치는 split-replicate 기준 **+0.033 [+0.003, +0.073]** (137개 audited states에서는 +0.033,
전체 686 states에서 +0.030~+0.041)이다.

**본문 문구를 이렇게 바꿔야 한다:**
- (X) "Results remain stable under sampled decoding" — 단일 draw 기준이라 과장.
- (O) greedy 정책에서 headroom은 max-of-noise가 아니며(matched-noise null +0.002, split 96% 유지),
  확률적 decoding으로 바꾸면 out-of-sample headroom은 +0.03 수준으로 줄어든다 →
  **realized utility가 정책 의존적이라는 논문의 주장과 정확히 일치**하므로, 축소가 아니라
  claim scope를 정확히 하는 방향으로 쓰는 게 맞다.

`review_response_values.md` 9절과 `manuscript_edits.tex` 블록 [7][8][9]를 이에 맞게 수정 완료.

---

# 최종 3항목 (A/B/C) — 실행 완료 (2026-08-20)

LaTeX: `results/review_tables/tables.tex`에 표 6종 (앞선 4종 + A/B 통합표 + C 표).

## 15. A. CE는 무작위보다 확실히 낫다 — 5 settings 전부 유의 ✅

Random은 **닫힌 형식**(후보집합 결과의 평균 = 균등 무작위 선택의 정확한 기댓값)으로 계산해
Monte-Carlo 잡음을 제거했다. 질문 단위 **paired** bootstrap.

| 세팅 | Random F1 | CE F1 | Oracle F1 | **CE−Random** | 95% CI |
|---|---|---|---|---|---|
| dev | 0.4083 | 0.4803 | 0.5624 | **+0.0720** | [+0.051, +0.095] |
| lockbox | 0.4013 | 0.4394 | 0.5560 | **+0.0381** | [+0.019, +0.057] |
| 2Wiki | 0.2541 | 0.2774 | 0.4197 | **+0.0233** | [+0.008, +0.040] |
| 14B | 0.4830 | 0.5398 | 0.6371 | **+0.0568** | [+0.037, +0.077] |
| MuSiQue | 0.1430 | 0.1658 | 0.2632 | **+0.0228** | [+0.010, +0.037] |

단일 무작위 draw 기준으로도 동일 (+0.024 ~ +0.071, 전부 CI가 0 제외).

→ "Your CE is basically random"에 대한 답: **아니다. 5개 세팅 전부에서 CE가 무작위 선택을
유의하게 앞서며, 그럼에도 측정 가능한 oracle headroom이 남는다.**

## 16. B. 강한/state-aware selector 잔여 headroom은 통계적으로 확실하다 ✅

### Oracle − selector (질문 단위 paired bootstrap)

| 세팅 | CE (MiniLM, q_t) | BGE (q_t) | **BGE (state-aware)** | Qwen listwise (state) |
|---|---|---|---|---|
| dev | +0.082 [.062,.104] | +0.099 [.075,.125] | **+0.087 [.065,.111]** | +0.095 [.071,.121] |
| lockbox | +0.117 [.092,.142] | +0.118 [.094,.144] | **+0.116 [.091,.141]** | +0.133 [.108,.160] |
| 2Wiki | +0.142 [.118,.168] | +0.135 [.111,.161] | **+0.136 [.112,.163]** | +0.141 [.115,.168] |
| 14B | +0.097 [.076,.120] | +0.096 [.075,.119] | **+0.090 [.071,.110]** | +0.100 [.080,.122] |
| MuSiQue | +0.097 [.080,.116] | +0.099 [.081,.118] | **+0.104 [.086,.125]** | +0.105 [.087,.125]|

**20개 칸 전부 CI가 0을 제외한다.** 잔여 headroom은 noise가 아니다.

### selector − CE 짝지은 차이 (강한 selector가 실제로 도움이 되는가)

| 세팅 | BGE(q_t) − CE | BGE(state) − CE |
|---|---|---|
| dev | **−0.017 [−0.033, −0.002]** | −0.005 [−0.022, +0.011] |
| lockbox | −0.002 [−0.015, +0.012] | +0.001 [−0.015, +0.017] |
| 2Wiki | +0.007 [−0.006, +0.021] | +0.006 [−0.010, +0.023] |
| 14B | +0.001 [−0.011, +0.013] | +0.008 [−0.008, +0.024] |
| MuSiQue | −0.002 [−0.010, +0.007] | −0.007 [−0.018, +0.004] |

10개 비교 중 9개가 0을 포함한다(유일한 예외는 dev의 BGE q_t가 CE보다 **나쁨**).
→ **더 강하게 만들거나 state를 줘도 선택 결과가 유의하게 달라지지 않는다.**

이것이 새 RQ2의 핵심 결과가 된다:
*Headroom persists under stronger and state-aware relevance selection.*

## 17. C. state-level gate — 불균형 아티팩트가 아니다 ✅

| 세팅 | prevalence | CE margin only | CE stats only | φ aggregates | **full gate** |
|---|---|---|---|---|---|
| dev | 0.140 | 0.531 | 0.547 | 0.578 | 0.599 |
| lockbox | 0.188 | 0.479 | 0.494 | 0.547 | **0.637** |
| 2Wiki | 0.200 | 0.500 | 0.541 | 0.538 | 0.527 |
| 14B | 0.167 | 0.515 | 0.535 | 0.568 | 0.560 |
| MuSiQue | 0.195 | 0.498 | 0.524 | 0.502 | 0.505 |

- **relevance confidence 단독은 사실상 chance** (CE margin 0.479~0.531). "CE가 스스로 실패를 안다"는
  가설은 성립하지 않는다.
- probe feature가 있는 두 세팅(dev, lockbox)이 가장 높다 → 신호의 상당 부분이 에이전트 불확실성에서
  온다. 단 이 두 세팅은 in-domain이기도 하다.

불균형 보정 지표 (full gate). AUC는 prevalence에 불변이고, AP의 무작위 기준선은 prevalence 자체다:

| 세팅 | prevalence | AP | **AP/prev (lift)** | prec@prev | lift |
|---|---|---|---|---|---|
| dev | 0.140 | 0.176 | 1.26 | 0.185 | 1.32 |
| lockbox | 0.188 | 0.281 | **1.49** | 0.287 | 1.53 |
| 2Wiki | 0.200 | 0.241 | 1.20 | 0.256 | 1.28 |
| 14B | 0.167 | 0.208 | 1.24 | 0.179 | 1.07 |
| MuSiQue | 0.195 | 0.208 | **1.07** | 0.187 | 0.96 |

→ **낮은 AUC는 class imbalance 탓이 아니다.** prevalence를 명시적으로 보정해도 lift는 1.0~1.5배에
그치고, MuSiQue에서는 prevalence 기준선과 사실상 동일하다.

**RQ3 서술 범위**: routing advantage 이야기는 쓰지 않는다(10절 플라시보). metric은 AUC / prevalence /
AP-lift / precision@prevalence까지만. 질문은
*Can states containing prospective headroom be identified before intervention?* 로 좁힌다.

---

# 최종 상태 요약

**확보된 방어 구조**

| RQ | 주장 | 근거 |
|---|---|---|
| RQ1 | headroom이 존재하며 max-of-noise가 아니다 | matched-noise null +0.002, split 96.1% 유지, cross-fit +0.111, representative stratum에서도 유지, CE ≫ random 5세팅 유의 |
| RQ2 | 더 강하고 state-aware한 relevance selector에서도 유지된다 | BGE(q_t)/BGE(state)/Qwen listwise 전부 20/20 칸 CI가 0 제외, selector−CE 차이 9/10이 0 포함 |
| RQ3 | headroom이 있는 state를 사전 식별하는 것은 어렵다 | AUC 0.505~0.637, CE confidence 단독은 chance, AP-lift 1.07~1.49 |

**반드시 원고에서 제거해야 할 기존 주장**

1. conflict-conditioned specialist 66~69% "conditional structure transfers" — invert-CE(오염 제거 시 100%)보다 낮음
2. anti-alignment signature — null-shuffle 안에 들어감 (11절)
3. brevity(doc_len) signature — 동일
4. "sampled decoding에서 결과 불변(+0.111)" — split 기준 +0.033 (14절)
5. oracle-condition routing 76.2% ceiling — trivial router 90.3%에 밀림 (4절)
6. oracle-state routing advantage — random placebo와 동일 (10절 하단, RQ3에서 서술 금지)

**반드시 고쳐야 할 정합성 문제**

- Opportunity 정의: 코드는 base 대비, 캡션은 CE-selected 대비 (3절). lockbox 29.3% vs 18.8%.
- supplementary의 pairwise label 정의(controlled → raw), "standardized features" → 이미 수정 완료.

---

# 리뷰어 v3 지적 (a)–(d) — 실행 완료 (2026-08-20)

## 18. (a) 강한 selector가 실제로 강하다는 독립 증거 ✅ — **최대 성과**

**측정**: retained 8개가 아니라 **full pool $\mathcal{C}_t$**(약 110개)에서 gold supporting
document에 대한 hit@1 / nDCG@10. pool은 계획 시점과 동일하게 재구성
(`fullpool_relevance.py`, BM25 5질의 + dense 2질의, seen 제외).

**재구성 정확성 검증**: 재구성 pool 크기가 논문 보고값과 정확히 일치 —
gate2 109.1±27.7 (논문 109.1±27.7), lockbox 108.2±28.7 (논문 108.2±28.7).

| 세팅 (pool에 gold 존재) | CE hit@1 | BGE q_t | **BGE state** | BGE q_t − CE | **BGE state − CE** |
|---|---|---|---|---|---|
| dev (68.6%) | 0.408 | 0.490 | **0.776** | +0.082 [+.038,+.128] | **+0.368 [+.298,+.438]** |
| lockbox (67.3%) | 0.320 | 0.422 | **0.765** | +0.102 [+.059,+.147] | **+0.445 [+.379,+.511]** |
| 2Wiki (55.9%) | 0.287 | 0.320 | **0.674** | +0.033 [+.001,+.064] | **+0.387 [+.313,+.457]** |
| 14B (68.7%) | 0.420 | 0.516 | **0.723** | +0.096 [+.051,+.141] | **+0.304 [+.235,+.371]** |
| MuSiQue (57.2%) | 0.246 | 0.292 | **0.444** | +0.046 [+.012,+.081] | **+0.198 [+.135,+.261]** |

nDCG@10 (random / CE / BGE q_t / BGE state):
dev .044/.458/.560/**.842** · lockbox .033/.387/.508/**.840** · 2Wiki .061/.362/.387/**.735** ·
14B .052/.481/.594/**.813** · MuSiQue .029/.364/.418/**.579**

- **10개 짝지은 비교 전부 CI가 0을 제외**한다. BGE는 명백히 더 강한 relevance 모델이고,
  state-aware BGE는 lockbox 기준 CE의 **hit@1을 2.4배**(0.320→0.765)로 올린다.
- 그런데 downstream F1은 CE와 구분되지 않고(9/10 CI가 0 포함), Oracle headroom도 그대로다
  (+0.1166 → +0.1156).

→ **논문의 최대 약점이 최강 증거로 뒤집힌다.** 주장이 "relevance는 불완전하다"에서
**"relevance 품질을 크게 올려도 downstream utility는 따라오지 않는다"**로 격상된다.

**정직하게 밝힐 것**: state-aware 질의에 $q_0$가 포함되고 gold는 $q_0$ 기준으로 주석되어 있어,
BGE-state의 우위 일부는 이 정의적 정렬에서 온다. 그래도 표준 지표상 더 나은 relevance 모델인 것은
사실이며, 핵심은 그 우위가 downstream으로 전이되지 않는다는 점이다.

## 19. (b) 확률적 디코딩 수치 복원 ✅

숨기지 말고 전부 쓸 것:

| lockbox | 값 |
|---|---|
| greedy 확증값 | +0.117 [+0.092, +0.142] |
| greedy split-replicate (137 audited states) | +0.130 (관측 +0.136의 **96.1%**) |
| greedy matched-noise null | **+0.0024** |
| sampled T=0.7, 같은 replicate에서 선택·평가 | +0.111 [+0.091, +0.133] |
| **sampled T=0.7, held-out replicate 평가** | **+0.033 [+0.003, +0.073]** |
| sampled matched-noise null | +0.0725 |
| 효과크기 ≥0.25 (Opp-base): greedy vs sampled 평균 | 27.8% vs 27.1% |

프레이밍(권장): PIU는 $U(d\mid s_t,\pi)$로 정의되고 greedy와 stochastic은 서로 다른 $\pi$다.
확률적 정책에서 realized headroom이 줄어드는 것은 결함이 아니라 **policy-dependence의 직접 실증**이다.
동시에 같은 검정이 greedy에서 null +0.0024를 준다는 사실이 **검정력의 양성 대조**가 된다.

## 20. (c) 학습된 neural state-level gate ✅ (결론 유지)

state-level 라벨 $Z(s_t)$로 트랜스포머 gate 2종을 직접 파인튜닝
(`rq3_neural_gate.py`, 입력 = 관측 state 텍스트 + 후보집합 정본 렌더링, gate3 2,368 states 학습).

| 세팅 | prev | CE stats | φ agg | full feature | MiniLM-L6 | DeBERTa-v3-base | **Best(사후선택)** |
|---|---|---|---|---|---|---|---|
| dev | .140 | .547 | .578 | **.599** | .516 | .478 | .599 |
| lockbox | .188 | .494 | .547 | **.637** | .500 | .525 | .637 |
| 2Wiki | .200 | .541 | .538 | .527 | .553 | **.567** | .567 |
| 14B | .167 | .535 | **.568** | .560 | .511 | .483 | .568 |
| MuSiQue | .195 | .524 | .502 | .505 | .486 | **.591** | .591 |

- (정정 2026-08-20: 위 표 14B 행의 full-feature 값은 .568이 아니라 **.560**이며, .568은 φ aggregates 값이다. 그림·JSON은 처음부터 옳았고 이 표만 오타였다.)
- **학습된 gate가 얕은 gate를 넘지 못한다.** 세팅별로 가장 유리한 family를 **사후 선택**해도
  AUC는 **0.567–0.637**에 그친다. 이건 낙관적 상한이므로 negative claim이 강화된다.
- **정직하게 밝힐 것**: DeBERTa는 학습 손실이 1.143→1.135로 거의 움직이지 않았다. 2,368 states는
  트랜스포머 gate에 작은 학습셋이며, "충분히 큰 state-level 학습셋에서는 달라질 수 있다"는
  가능성을 limitation에 남겨야 한다. 다만 이는 문제의 어려움을 과소평가한 게 아니라
  **현재 규모에서 학습 가능한 신호가 약하다**는 뜻이다.
- Figure 2(b)에 "Fine-tuned transformer" 열을 추가해 그림에서도 바로 보이게 했다.

## 21. (d) opportunity 정의 분리 ✅ — **재계산 아님, 명명 문제**

확인 결과 v3의 수치는 **틀리지 않았다.** 29.3%→27.8%는 Opp-base로 정확하고, Table 5의 18.8%는
Opp-CE다. 같은 이름을 쓴 서로 다른 두 양이었다. 두 이름으로 분리하고 둘 다 싣는 것으로 해결된다.

| 세팅 | Opp-base >0 | Opp-base ≥.25 | Opp-CE >0 | Opp-CE ≥.25 |
|---|---|---|---|---|
| dev | 30.1% | 26.4% | 14.0% | 12.4% |
| lockbox | 29.3% | 27.8% | 18.8% | 16.9% |
| 2Wiki | 26.4% | 24.8% | 20.0% | 17.6% |
| 14B | 32.1% | 29.1% | 16.7% | 14.3% |
| MuSiQue | 21.8% | 18.8% | 19.5% | 16.1% |

stratum별 (representative / fail-enriched):

| 세팅 | rep base | rep CE | fail base | fail CE |
|---|---|---|---|---|
| dev | 18.5% | 9.1% | 40.5% | 18.4% |
| lockbox | 22.5% | 18.9% | 35.7% | 18.7% |
| 2Wiki | 28.4% | 22.3% | 24.3% | 17.8% |
| 14B | 25.6% | 14.1% | 38.0% | 19.1% |
| MuSiQue | 19.6% | 21.3% | 24.0% | 17.7% |

**원고 조치**: Fig.1 캡션·Effect magnitude의 수치는 그대로 두되 이름을 `Opp-base`로 명시하고,
Table 5의 prevalence는 `Opp-CE`로 명시. 새 표(`tab:oppdefs`)로 둘을 나란히 싣는다.

---

# 갱신된 방어 구조

| RQ | 주장 | 근거 |
|---|---|---|
| RQ1 | headroom 존재, max-of-noise 아님 | matched-noise null +0.0024, split 96.1%, cross-fit +0.111, representative 유지, CE≫random 5세팅 |
| RQ2 | **relevance 품질을 크게 올려도 downstream은 안 따라온다** | BGE-state가 gold hit@1 +0.198~+0.445로 우월(10/10 유의)한데 downstream F1은 CE와 무차별(9/10 무차별), Oracle headroom 불변 |
| RQ3 | headroom state의 사전 식별이 어렵다 | 얕은 gate + 학습된 트랜스포머 2종 전부 실패, 사후 최적 선택해도 AUC ≤ 0.637 |

---

# 리뷰어 W1 / W6 / 처방 — 실행 완료 (2026-08-20)

## 22. W1 trajectory 분해 ✅ — **논문에 유리. novelty 방어 성공**

`d_PIU`와 `d_CE` 두 arm이 개입 후 무엇을 했는지로 state를 분해
(`w1_trajectory_decomposition.py`). 개입 후 추가 검색 스텝 분포(lockbox):
0회 366 / 1회 210 / 2회 82 / 3회 28 → **47%의 state에서 최소 한 arm이 추가 검색을 한다.**
(본문의 "평균 0.62 스텝"은 8개 후보 arm 전체 평균이라 더 작게 보인다. 비교에 실제로 들어가는
두 arm 기준으로는 훨씬 크다 — 이 구분을 본문에 명시할 것.)

| 세팅 | no_search | all | any_search | **query_changed** |
|---|---|---|---|---|
| dev | +0.051 [.031,.072] | +0.082 | +0.133 [.095,.175] | **+0.177 [.122,.237]** |
| **lockbox** | +0.079 [.054,.106] | +0.117 | +0.169 [.130,.211] | **+0.311 [.244,.380]** |
| 2Wiki | +0.085 [.058,.114] | +0.142 | +0.190 [.154,.229] | **+0.274 [.213,.340]** |
| 14B | +0.050 [.030,.072] | +0.097 | +0.143 [.112,.177] | **+0.223 [.177,.273]** |
| MuSiQue | +0.086 [.061,.113] | +0.097 | +0.111 [.089,.134] | **+0.131 [.100,.163]** |

- **5개 세팅 전부 no_search < all < any_search < query_changed 순서.** lockbox에서
  query_changed subset의 headroom(+0.311)은 no_search subset(+0.079)의 **약 4배**다.
- 동시에 no_search subset에서도 CI가 0을 제외한다 → reader-context 효과도 실재한다.
- 따라서 정직한 서술: **두 메커니즘이 모두 있고, headroom은 궤적이 실제로 바뀐 state에 집중된다.**
  "measurable headroom persists — and roughly triples — when the intervention changes the
  agent's subsequent search" 라고 쓸 수 있다. trajectory-level 프레이밍을 유지할 근거가 생겼다.

## 23. W6 oracle-vs-control 재계산 ✅ — **controlled 지표가 살아난다**

전체 후보 평균이 아니라 **각 규칙이 실제로 고른 후보**의 controlled PIU (`w6_oracle_vs_control.py`):

| 세팅 | anti-oracle | random | **CE** | **oracle** | **oracle−CE (paired)** |
|---|---|---|---|---|---|
| dev | −0.092 | +0.011 | **+0.085 [.059,.113]** | **+0.160 [.130,.192]** | +0.075 [.052,.099] |
| lockbox | −0.100 | −0.003 | **+0.044 [.018,.069]** | **+0.152 [.122,.183]** | **+0.109 [.077,.142]** |
| 2Wiki | −0.102 | +0.011 | +0.025 [.002,.047] | **+0.164 [.137,.190]** | +0.138 [.110,.168] |
| 14B | −0.085 | +0.021 | **+0.066 [.040,.091]** | **+0.162 [.133,.192]** | +0.097 [.069,.126] |
| MuSiQue | −0.072 | +0.001 | +0.030 [.015,.046] | **+0.113 [.094,.133]** | +0.083 [.065,.102] |

- 논문이 보고한 "+0.0060, CI가 0 포함"은 정확히 **random 행**에 해당한다. 모든 후보를 평균하면
  0 근처인 게 당연하다는 리뷰어 지적이 맞다.
- **CE 행은 선택 편향이 없다** (CE는 outcome을 보지 않음). 그런데도 5세팅 전부 유의하게 양수 →
  개입 장치가 실제로 작동한다는 깨끗한 검증이다.
- **oracle−CE paired contrast(+0.109, lockbox)가 raw headroom(+0.117)과 거의 일치한다** →
  결론이 raw/controlled 추정량 선택에 의존하지 않는다(수렴 타당도).
- oracle 행은 raw로 선택하고 controlled로 평가하므로 선택 효과가 남는다. random(≈0)과
  anti-oracle(강한 음수)이 양쪽에서 경계를 준다는 점을 함께 쓸 것.

## 24. 처방 — budgeted lookahead (proof-of-concept)

CE 상위 k개만 rollout했을 때 회수되는 headroom (`lookahead_budget.py`).

### (i) Oracle-k — 상한 (rollout을 채점할 수 있다고 가정)

| 세팅 | k=2 회수율 | k=3 | k=4 | k=6 |
|---|---|---|---|---|
| dev | 30.9% | 64.0% | 76.6% | 89.7% |
| lockbox | **46.4%** | **65.4%** | 76.0% | 93.5% |
| 2Wiki | 37.3% | 60.8% | 73.6% | 95.2% |
| 14B | 46.3% | 70.6% | 80.0% | 92.9% |
| MuSiQue | 32.2% | 54.2% | 68.2% | 85.8% |

→ **CE 상위 2~3개만 rollout해도 전체 headroom의 31~46% / 54~71%가 그 안에 있다.**
headroom이 CE 상위권에 집중돼 있다는 뜻이라 실무적으로 중요하다.

### (ii) 배포 가능한 gold-free 규칙 2종은 실패

| 규칙 | 결과 |
|---|---|
| 다수결 투표 (k개 continuation의 답) | k=2는 CE와 동일(동점→CE), k=3 이상은 **CE보다 나빠짐** (−3%~−35% 회수율) |
| 조기 종료 우선 (추가 검색 없이 답한 후보) | 전 세팅에서 ±0.005 이내, 일관된 이득 없음 |

### (iii) 왜 실패하는가 — 원인 규명 ✅

opportunity state에서 후보 8개 continuation 중 **다수답(modal answer)** 인지:

| 세팅 | oracle 후보가 modal | CE 후보가 modal |
|---|---|---|
| dev | **24.7%** | 45.7% |
| lockbox | **28.7%** | 34.1% |
| 2Wiki | **23.8%** | 43.9% |
| 14B | **24.3%** | 47.9% |
| MuSiQue | **22.5%** | 36.9% |

→ **유용한 후보의 답은 대체로 소수답이다.** 따라서 self-consistency/다수결 계열 집계는
utility와 **구조적으로 역상관**이다. 이건 휴리스틱 하나가 실패했다는 보고보다 훨씬 강한 진술이다.

### 정리된 constructive message

> 병목은 rollout 예산이 아니라 **gold 없이 rollout을 채점하는 방법**이다.
> CE 상위 2~3개 rollout 안에 headroom의 절반 이상이 들어 있지만, 자연스러운 gold-free 집계
> 두 가지(합의 투표, 조기 종료)는 모두 실패하며, 그 이유는 유용한 continuation이 대체로
> 소수답이기 때문이다.

이 형태면 RQ3의 null이 **"어디가 어려운지 특정한 결과"** 로 승격되고, "처방이 없다"는 W5 공격도
상당 부분 약해진다. diagnostic paper 정체성은 유지된다.

## 25. W3 보충 — stochastic을 lockbox 전체로 확장한 값

리뷰어가 "137개에서만 측정"이라 지적했으나, **sampled decoding은 lockbox 686 states 전부에서
이미 측정돼 있다** (`results/decoding_sensitivity`, 6,173 arms × 3 replicates).
held-out replicate 평가 기준 fold별 **+0.0298 / +0.0281 / +0.0407** (전체 686 states).
137개는 greedy replay와 **동일 state에서 맞비교**하기 위한 부분집합이었을 뿐이다.
본문에 "전체 lockbox에서 측정했고, 137개는 greedy와의 matched 비교용"이라고 명시하면
이 지적은 사실관계로 해소된다.

---

# A급 / B급 — 실행 완료 (2026-08-21)

## 26. A급: prospective continuation scorer — **약한 양성 (proof-of-concept 수준)**

CE top-3의 기존 rollout에서 **gold-free 관측 신호만**으로 continuation을 채점하는 pairwise
로지스틱 (`continuation_scorer.py`, `continuation_scorer_v2.py`). feature 12종: 상태 내 z-CE 점수,
추가 검색 수, 답변 종료 여부, 즉시 종료, 토큰/턴, 답 길이, 빈 답, **답이 주입 문서에 지지되는 정도**,
**답이 개입 후 검색 근거에 지지되는 정도**, base 답과 동일 여부, top-3 내 modal 여부.
F1은 학습 라벨로만 사용, feature에는 절대 미포함. gate3 학습(1,564 decided pairs) 후 동결.

**v1 (자유 선택): 실패.** dev −0.022(유의 음수), lockbox −0.000, 방향 비일관.

**v2 (CE-sticky, override margin을 dev에서만 튜닝 후 동결, margin=1.0):**

| 세팅 | overrides | gain vs CE | Oracle@3 | 회수율 | **pairwise acc** |
|---|---|---|---|---|---|
| dev (튜닝셋) | 12/579 | +0.0011 [−.003,+.005] | +0.053 | — | 0.695 |
| **lockbox** | 13/686 | **+0.0035 [+.0007,+.0071]** | +0.076 | 4.6% | 0.631 |
| **2Wiki** | 39/819 | **+0.0075 [+.0017,+.0143]** | +0.086 | 8.7% | 0.595 |
| **14B** | 49/838 | **+0.0096 [+.0025,+.0172]** | +0.069 | 14.0% | 0.754 |
| MuSiQue | 58/959 | +0.0004 [−.004,+.005] | +0.053 | 0.8% | 0.544 |

**판정 (리뷰어 기준 대비 정직하게):**
- 통과한 것: lockbox 유의 양수 + 4개 확증 세팅 전부 **음수 없음** + 3/4 유의 양수.
- 못 미친 것: 리뷰어가 예시한 +0.02(25–30% 회수)에는 크게 못 미침 (+0.004~+0.010, 5–14%).
- **의외의 강한 부분은 pairwise accuracy다**: decided continuation pair에서 0.54–0.75.
  개입 전 gate의 AUC 0.505–0.637과 대비하면, **rollout 후 관측 신호는 개입 전 신호보다
  훨씬 정보량이 크다.** 이 대비가 Figure 3의 "병목은 채점"과 정확히 이어진다.
- **권장 배치**: main headline 아님. RQ3 뒤 짧은 proof-of-concept 문단 + supplement 표
  (`tab:scorer`). 문구는 "small but statistically significant gains, never negative,
  with a 2–7% override rate" 수준으로 절제. 과장하면 W5를 막으려다 새 공격면이 열린다.

## 27. B급: oracle winner 특성 분석 — **강함. mechanistic finding 확보**

### (1) 실패 state에서 oracle winner의 CE rank (state-count 관점, Fig.3c의 보완)

| 세팅 | n | median rank | **top-2** | **top-3** | rank 5+ |
|---|---|---|---|---|---|
| dev | 81 | 4 | 17.3% | 38.3% | 34/81 |
| **lockbox** | 129 | **3** | **33.3%** | **50.4%** | 42/129 |
| 2Wiki | 164 | 4 | 23.8% | 47.0% | 58/164 |
| 14B | 140 | 4 | 30.7% | 45.7% | 54/140 |
| MuSiQue | 187 | 4 | 17.1% | 35.3% | 84/187 |

→ headroom-mass 관점(Fig.3c: top-3에 54–71%)과 state-count 관점(top-3에 35–50%)이 둘 다 확보됨.
두 질문이 다르다는 리뷰어 지적 그대로 반영 가능.

### (2) φ(d_PIU) − φ(d_CE), **outcome-permutation null 대조** (300회)

핵심: 나이브하게 보면 winner가 CE보다 query_align이 낮다(−0.08~−0.14). 그러나 winner는 정의상
CE argmax가 아니므로 이 차이의 대부분은 기계적이다. null(state 내 결과 permute)과 비교하면
**방향이 뒤집힌다**:

| feature | null 밖 세팅 수 | 방향 (null 대비) |
|---|---|---|
| doc_len | **5/5** | ↑ (null보다 덜 짧다 — 즉 chance 대안보다 길다) |
| query_align | **4/5** | ↑ (chance 대안보다 질의 정렬이 높다) |
| gap_coverage | **4/5** | ↑ |
| q0_align | 3/5 | ↑ |
| redundancy | 3/5 | **↓ (chance 대안보다 소비 근거와 덜 중복)** |
| novelty | 2/5 | ↑ (덜 novel — "exotic 문서 아님"과 일관) |
| entity_gain | 0/5 | 신호 없음 |

**정직한 mechanistic 문장:**

> Utility winners are not exotic, low-relevance documents. Relative to a
> conditioning-matched null, they are *more* relevance-like — better aligned
> with the current query and question, longer, with higher coverage of
> unresolved question terms — and *less redundant* with already-consumed
> evidence than chance alternatives. This explains why they concentrate near
> the top of the relevance ranking (35–50% within CE top-3) yet are missed by
> it: relevance nearly finds them, and what it fails to price is redundancy
> with the evidence the agent already holds.

- 이 분석은 기존 시그니처 주장을 죽였던 것과 **같은 null 방법론**을 통과했다. 방어 가능.
- 옛 "anti-alignment" 서사와 정반대 방향이므로, 본문 어디에도 낡은 방향의 문장이 남지 않게 주의.
- 표: `tab:winnerrank`, `tab:winnerchar` (LaTeX 자동 생성, 총 15개 표).

### 종합 스토리 (리뷰어 제안 구조로)

Relevance misses useful evidence → stronger relevance still misses it →
the winners sit near the top of the ranking and differ mainly by redundancy
with consumed evidence → most of their value is within a top-3 rollout budget →
post-rollout observable signals rank continuations far better than
pre-intervention signals (0.54–0.75 pairwise vs 0.51–0.64 AUC), and a
conservative scorer already converts this into small significant gains.

---

# 28. 선택공간 일치성 (C_t vs D_t) — 실행 완료 (2026-08-21)

리뷰어의 "남은 가장 강한 objection": BGE의 relevance 우위는 full pool C_t(~110개)에서 보였는데
downstream 개입은 D_t(≤8개)에서만 이루어지며, BGE의 full-pool argmax가 D_t에 있다는 보장이 없다.
**분석 A·B(기존 데이터) + 신규 rollout C로 완전히 닫음.**

스크립트: `choice_space_consistency.py`(예비), `fullpool_relevance.py --ce-full-text --doc-chars 600`
(정확 설정 재채점, `results/selector_relevance_exact/`), `expand_choice_space.py`(신규 arms),
`analyze_choice_space.py`(최종). 결과: `results/choice_space/report_expanded.json`.

## 방법론 주의 — 절단 아티팩트 (필수 숙지)

첫 계산(500자 절단)에서 CE retention이 0.94~0.98로 나왔으나, 이는 **논문의 채점 설정(CE 전문,
BGE 600자)과 다른 절단** 때문이었다. 500자 절단은 CE argmax를 2~7% 뒤집는다.
**동일 설정으로 재채점하면 CE retention = 3,880/3,881 = 정확히 1.000** (논문의 "예외 0"과 일치).
유일한 예외 1건(MuSiQue 3hop1__222979_40769_64047::s5)은 pool 재구성 시 dense 검색의 한계 순위
차이로 원본 pool에 없던 문서가 유입된 재구성 아티팩트이며, 원본 파이프라인과 무관.
→ **18절의 full-pool 표(500자 기준)는 supplement에 쓸 경우 정확 설정 수치로 교체할 것.**
정확 설정에서도 정성적 결론은 동일 (BGE-state hit@1 우위 유지).

## A. Full-pool argmax retention (정확 설정)

| 세팅 | CE | BGE(q_t) | BGE-state |
|---|---|---|---|
| dev | 1.000 | 0.668 | 0.366 |
| lockbox | 1.000 | 0.650 | 0.318 |
| 2Wiki | 1.000 | 0.578 | 0.249 |
| 14B | 1.000 | 0.706 | 0.372 |
| MuSiQue | 0.999 | 0.602 | 0.264 |

→ 리뷰어 우려가 사실로 확인됨: BGE-state의 full-pool 선택은 25~37%만 D_t에 있었다.
**그래서 C의 rollout이 필요했다.**

## B. Hit@1 over D_t — relevance 우위는 정확히 그 선택공간에서도 성립

gold가 D_t에 있는 state 한정 (커버리지 24~43%):

| 세팅 | CE | BGE | BGE-state | **BGE-state − CE (paired)** |
|---|---|---|---|---|
| dev | 0.753 | 0.846 | 0.900 | **+0.147 [+.074,+.223]** |
| lockbox | 0.719 | 0.801 | 0.903 | **+0.184 [+.100,+.268]** |
| 2Wiki | 0.707 | 0.796 | 0.790 | **+0.082 [+.007,+.158]** |
| 14B | 0.749 | 0.819 | 0.852 | **+0.103 [+.038,+.172]** |
| MuSiQue | 0.555 | 0.671 | 0.685 | **+0.130 [+.044,+.216]** |

**5/5 세팅 paired CI가 0 제외.** "C_t에서만 강했다"는 반론이 데이터로 기각된다.

## C. 신규 rollout — full-pool argmax를 실제로 개입시킨 결과

D_t에 없던 BGE/BGE-state full-pool argmax **3,861개 arm**을 동결 greedy 정책으로 rollout
(세팅당 28~91초). 이제 각 selector가 **자신의 진짜 full-pool 선택**으로 평가된다. missing 0.

| 세팅 | BGE(full) − CE | BGE-state(full) − CE | Oracle(D_t) − CE | Oracle(확장) − CE |
|---|---|---|---|---|
| dev | −0.019 [−.042,+.004] | +0.012 [−.016,+.043] | +0.082 | **+0.125** |
| lockbox | +0.008 [−.010,+.026] | +0.024 [−.003,+.051] | +0.117 | **+0.151** |
| 2Wiki | +0.010 [−.006,+.025] | +0.018 [−.007,+.043] | +0.142 | **+0.168** |
| 14B | −0.010 [−.029,+.010] | −0.009 [−.034,+.016] | +0.097 | **+0.136** |
| MuSiQue | −0.001 [−.014,+.012] | −0.000 [−.017,+.017] | +0.097 | **+0.111** |

- **full-pool 선택을 허용해도 10개 비교 전부 CI가 0을 포함한다.** hit@1을 최대 2.4배 올린
  selector가 자신의 최선 선택으로도 downstream에서 CE와 구분되지 않는다.
- 확장 oracle은 +0.111~+0.168로 **더 커진다** (더 많은 후보의 max이므로 기계적으로 당연하나,
  강한 relevance가 가리키는 문서들 위에서도 headroom이 여전히 크다는 뜻).

## 정직하게 함께 쓸 것

1. 신규 arms는 개정 단계 post-hoc rollout이다 (lockbox 개봉 후). RQ1 endpoint 불변, 표기 필수.
2. 확장 oracle을 새 headline으로 쓰지 말 것 — headroom 주장은 기존 Oracle(D_t) 유지, 확장은
   맥락으로만. (max 대상이 커지면 oracle은 기계적으로 커진다.)
3. B의 hit@1이 full pool(0.25~0.44)보다 높은 이유는 선택지가 8개뿐이라서다. 두 수치를 섞지 말 것.

## RQ2 최종 논증 사슬 (이제 완결)

1. BGE-state는 full pool에서 CE보다 훨씬 강한 relevance 모델 (hit@1 2.4배, 18절)
2. 그 우위는 downstream이 이루어지는 **정확한 선택공간 D_t에서도 성립** (B, 5/5 유의)
3. full-pool 최선 선택을 실제 개입으로 평가해도 downstream 이득 없음 (C, 10/10 무차별)
4. 같은 확장 공간에서 oracle headroom은 +0.11~+0.17

→ "relevance 개선과 downstream 비교가 다른 선택공간에서 이뤄졌다"는 objection은
논증이 아니라 **실측으로** 닫혔다.

표: `tab:choicespace`, `tab:expanded` (총 17개, 자동 생성).

---

# 29. Myopic utility vs Full PIU — 실행 완료 (2026-08-23, HotpotQA lockbox)

같은 frozen state·같은 후보·같은 동결 greedy 정책. **Full**: 주입 → 정상 search continuation →
답. **Myopic**: 주입 → 추가 retrieval 금지(harness의 forced-answer 블록) → 즉시 답.
6,173 arms(686 base + 5,487 개입), 69초. 스크립트 `myopic_vs_full.py` + `analyze_myopic.py`,
결과 `results/myopic/report.json`. 금지에도 검색을 시도한 비율 2.7% (빈 답으로 처리, 확증 프로토콜과 동일).

| 지표 | 값 |
|---|---|
| (state, candidate) 수준 상관 | Pearson 0.661 / Spearman 0.652 |
| state 내 Spearman 평균 (분산 있는 159 state) | **0.392** |
| myopic 최댓값 **동점** state 비율 | **84.1%** (opportunity state에서 69.0%) |
| 승자 any-max 일치 (관대) | 96.4% 전체 / 85.3% opportunity |
| 승자 strict 일치 (CE tie-break, opportunity state) | **27.1%** |
| Full oracle − CE | +0.117 [+0.092, +0.142] |
| Myopic oracle − CE (myopic 결과 기준) | +0.083 [+0.063, +0.104] |
| **Full headroom, myopic→CE tie-break로 선택 시** | **+0.015 [+0.003, +0.027]** (12.7% 회수) |
| Full headroom, myopic 최댓값 집합 내 최선 (상한) | +0.093 [+0.071, +0.116] (80%) |
| base F1: full / myopic | 0.327 / 0.278 |

**해석 (정직하게):**
- myopic utility는 full PIU와 **중간 정도 상관**(ρ≈0.65)하지만 **너무 거칠다** — 즉시 답하면 많은
  후보가 같은 답을 내서 84%의 state에서 최댓값이 동점이다.
- 동점 집합 *안에는* full headroom의 80%가 들어 있다(상한 +0.093). 그러나 myopic은 그 안을
  구분하지 못하므로 selector로 쓰면 **12.7%만** 회수된다(+0.015, 유의하나 작음).
- opportunity state에서 strict 승자 일치는 27%뿐 → **"문서를 읽기만 하면 되는" reader 효과로는
  PIU 순위를 재구성할 수 없다.** W1 분해(query_changed에서 headroom 4배)와 같은 방향.
- myopic oracle 자체 headroom(+0.083)이 full(+0.117)보다 작다 → continuation이 가치를 *만든다*.

**논문 문장 후보**: "Myopic (retrieval-free) utility correlates only moderately with full PIU
(ρ=0.65) and is coarse: its maximum is tied in 84% of states. Used as a selector with CE
tie-breaking it recovers 13% of the full headroom, and the set of myopic-best candidates contains
80% of it — the continuation, not the read, determines which of them is useful."

---

# 30. Cross-family replication — microsoft/phi-4 ✅ (2026-08-24)

**모델 선택**: Mistral-7B-v0.3(사전등록 FAIL, malformed 42.2%)과 Llama-3.1-8B(FAIL, 재측정 금지)는
불가. **microsoft/phi-4 (14B, Microsoft 계열, MIT, system role 지원)** 선택.

**① Compliance 스모크 (사전등록 프로토콜, dev offset 1700, n=50):** **PASS**
- valid final answer 92.0% (기준 ≥90) / malformed search **0.0%** (≤5) / token-limit 2.0% (≤10) /
  hallucinated leakage **0** (=0) / 참고 F1 0.563 (기준 아님)
- Mistral 42.2%, Llama 5.9%와 대조적으로 여유 있게 통과.

**② 복제 (동결 프로토콜, HotpotQA representative 350문항, failure enrichment 없음):**
base 400(EM **0.422** — Qwen-14B 0.427과 대등) → 860 states → 7,730 arms (base+I2_add;
control은 확증 지표가 raw 정의라 생략) → 6,870 interventions. 총 GPU 시간 약 45분.

| 지표 | Phi-4 | (Qwen 5세팅 범위) |
|---|---|---|
| **Oracle−CE ΔF1** | **+0.081 [+0.063, +0.102]** | +0.082 ~ +0.142 |
| Opp-Base / Opp-CE | 25.0% / 15.8% | 21.8~32.1% / 14.0~20.0% |
| CE−Random (paired) | +0.048 [+0.029, +0.067] | +0.023 ~ +0.072 |
| Oracle−BGE(q_t) | +0.071 [+0.053, +0.089] | 유의 유지 |
| Oracle−BGE(state) | +0.078 [+0.060, +0.099] | 유의 유지 |
| BGE(q_t)−CE (paired) | **+0.011 [+0.002, +0.021]** | 9/10 무차별 |
| BGE(state)−CE (paired) | +0.003 [−0.010, +0.016] | 무차별 |

- **현상이 다른 model family에서 완전히 재현된다**: headroom 유의, 크기가 Qwen 범위 하단과 일치,
  representative 표본만으로, CE는 random보다 유의하게 나음, 강한 selector도 headroom을 못 닫음.
- 특기: BGE(q_t)−CE가 여기서 처음으로 **작지만 유의한 양수**(+0.011). 정직하게 보고할 것 —
  "stronger relevance가 약간의 이득을 줄 수는 있으나 Oracle 대비 headroom(+0.071)은 그대로"가
  정확한 서술이다. 강한 selector 무용론이 아니라 headroom 지속이 주장이므로 오히려 무해하다.
- **표기 필수**: post-hoc cross-family replication (모든 분석 후 실행, 사전등록 아님).
  단 compliance 기준·프로토콜·τ·추정량은 동결본 그대로.
- **Limitation 업그레이드**: "Cross-family validation was constrained by tool-protocol
  compliance" → "…; a post-hoc replication with a compliant 14B agent from a different
  family (phi-4) reproduces the headroom (+0.081 [0.063, 0.102]) on a representative sample."
- 산출물: `results/xfamily_phi4/XFAMILY_REPORT.json`, 드라이버 `xfamily_phi4_driver.sh`,
  스모크 로그 `logs/smoke_phi4.log`. 표 `tab:xfamily`.

(29절 myopic 결과 포함, 표 `tab:myopic` — 총 19개 표 자동 생성.)
