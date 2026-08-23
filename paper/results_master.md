# Results Master — 전 실험 확정 수치 (집필용 단일 소스)

작성: 2026-08-13 · 모든 수치는 각 results/ 디렉토리의 보고서에서 전사. 수정 시 반드시 원본 재확인. MuSiQue(R19) 결과는 도착 시 추가.

## A. 일반화 사다리 — H1 계열 (Oracle−CE headroom, 질문 단위 paired bootstrap)

| 세팅 | n(질문) | states | Oracle−CE ΔF1 | 95% CI | Opportunity |
|---|---|---|---|---|---|
| Hotpot dev (Gate 2, 사전등록) | 297 | 579 | +0.082 | [+0.062, +0.104] | 30.1% (대표 18.5 / 실패 40.5) |
| Hotpot **lockbox** (1회 개봉) | 350 | 686 | **+0.117** | [+0.092, +0.142] | 29.3% |
| **2Wiki** (R17) | 350 | 819 | **+0.142** | [+0.117, +0.168] | 26.4% |
| **Qwen-14B** (R18, cross-scale) | 350 | 838 | **+0.097** | [+0.076, +0.120] | **32.1%** |
| **MuSiQue** (R19) | 349 | 959 | **+0.097** | [+0.080, +0.116] | 21.8% |

- 표현 규칙: oracle은 구조상 CE 이하 불가 → "measurable actionable headroom exists beyond CE" (승률 아님).
- 14B base EM 0.427 > 7B 0.378 인데 opportunity 유지 → weak-agent artifact 기각.

## B. 조건부 구조 전이 — H2a 계열 (frozen conflicting-specialist, gate3 학습)

| 평가 대상 | acc | n | 이항 p |
|---|---|---|---|
| Hotpot lockbox | 69.2% | 1,639 | ~6e-56 |
| 2Wiki | 66.0% | 1,684 | ~4e-40 |
| Qwen-14B 궤적 | 66.7% | 1,930 | ~1e-49 |
| MuSiQue | 66.6% | 2,014 | ~3e-51 |

## C. Robust signature — conflicting 조건 재적합 계수 (방향)

| 세팅 | query_align | doc_len | redundancy | novelty |
|---|---|---|---|---|
| Hotpot dev | −2.54 (1.00) | −3.37 (1.00) | +1.06 (1.00) | +0.50 (0.96) |
| lockbox | −2.45 (1.00) | −2.73 (1.00) | −1.41 (0.998) | −0.21 (0.77) |
| 2Wiki | −1.91 (1.00) | −3.82 (1.00) | +1.14 (0.99) | +1.74 (1.00) |
| 14B | −2.93 (1.00) | −2.16 (1.00) | −1.79 (1.00) | −0.29 (0.78) |
| MuSiQue | −2.19 (1.00) | −2.00 (1.00) | +1.76 (1.00) | +1.18 (1.00) |

- **Confirmatory 2종 (anti-alignment, brevity): 전 세팅 재현.** redundancy/novelty: pool-dependent (exploratory).

## D. 식별 한계 (H3, Hotpot lockbox 기준)

- Specialists ceiling: dev 76.0% / lockbox 76.2% / 2Wiki 71.2% / 14B 75.2% — ~75% 안정
- Deployable gate (동결): AUC 앙상블 dev 0.691 → lockbox 0.641; **mixture recovery −0.02 (≈0)**
- Gate 캠페인 전체(lexical → +probe/CE분산 → +semantic → transformer): AUC 0.61–0.69 수렴, routing 회수 0
- Predictor: v1 57.0%, v2 최고 59.3% (전체) / CE-disagreement 39–47% / CE 단독 53.1% / CE+PIU 62.1%

## E. Agent overlap (R18″, exploratory)

- ρ(U_7B, U_14B) = 0.101 (p=0.015, n=579 (question,doc)) — **doc 단위 utility는 agent-specific, conditional ordering 구조는 전이** (B와 대비가 핵심 서사)

## F. Compliance audit (appendix)

| Agent | malformed | valid ans | token-limit | leakage | 판정 |
|---|---|---|---|---|---|
| Qwen2.5-7B (주) | (native 준수) | — | — | — | 기준 에이전트 |
| Mistral-7B-v0.3 | **42.2%** | 100% | 0% | 0 | 제외 |
| Llama-3.1-8B | **5.9%** (>5.0) | 96.7% | 3.3% | 0 | near-boundary 제외 (재측정 금지) |
| Qwen2.5-14B | **0.0%** | 93.3% | 6.7% | 0 | 통과 |

- Harness 무결성: R18′ 패치 2종, Qwen no-op 99% 기준 FAIL(94%) → same-code control 76% (설정 변동 포함) + mechanism activation 0 → "no evidence of behavioral effect" (methods 문구 R18′-control에 확정)

## G. 부수 발견 (secondary)

- Gold-but-Non-Useful: main 74.2% ↔ stress 37.5% (gold 유용성도 state-dependent), Gate2 62.9%
- Context pollution: non-gold 문서 제거가 F1 +0.057 (I1, n=330)
- 토큰 회계: rollout 비용의 96.2%가 prompt
- U_search 케이스: terminates 7.1%→12.9% (스트레스) — utility에 "검색 종료 유발" 포함
- Conditional: opportunity가 base-실패 state에 집중 (37.5% vs 0.8%)
- Pairwise 라벨: consistency 97.8~99.0%, preference strength 이봉분포, τ 민감도 100% 방향 일치

## H. 주요 provenance

- git: 6bc80db(lockbox 봉인) → 020ab24(개봉) → a7d75c1(R17) → 4083cfc(2Wiki) → 0ff8a1a(R18′) → b542d84(control) → 9c62b61(R19) → 94ed92b(14B) → 37e6235(R20)
- 사전 등록 체계: R10(τ, primary), R16(lockbox H1-3), R17/R18/R19(replications), R14(predictor 판정), R20(종료 선언)
## I. R21 리뷰 방어 (2026-08-13)

- **② 질문 클러스터 H2**: lockbox 66.4% CI[61.8,70.9] / 2Wiki 63.3% CI[58.5,67.9] / 14B 64.9% CI[60.4,69.0] — 전부 perm-p<0.0005
- **③ Standard-only Oracle** (bm25/dense/ce 후보만): lockbox +0.078 / 2Wiki +0.083 / 14B +0.071 — 전부 CI 유의. LOO: 단일 유형 제거로 gain 거의 불변 (최대 변화 ce_top 제거 시 −0.018)
- **④ 효과 크기**: opportunity의 ~95%가 ΔF1≥0.25 (3개 세팅 공통) — 미세 효과 아님
- **①-aux 교환가능성 null**: CE는 pool 내 무작위 선택보다 유의하게 나음 (obs +0.117 < null +0.155 CI[0.141,0.169]) — CE는 진지한 베이스라인
- **① Cross-fitted Oracle (winner's curse 방어, 최종)**: Gain_CF **+0.111 CI[+0.080,+0.144]** vs naive +0.114 (차이 0.4pp) — 선택·평가 분리 후에도 headroom 유지. A→B +0.117 / B→A +0.104. n=150 (dev offset 1800 신규 세그먼트)

## J. 집필 중 확인된 정정 사항 (2026-08-15)

- **Matched control의 "미소비 문서" 조건은 코드상 보장되지 않음.** `match_distractors`는 BM25(q_t) top-100에서 후보·gold·사용된 후보만 제외하고 **이미 소비된(seen) 문서는 제외하지 않음**. 실측: 전체 control 중 이미 소비된 문서 비율 = gate2 9.0% / lockbox 9.1% / 2Wiki 8.4% / 14B 9.9% / MuSiQue 10.0%. 
  - **확증 수치 영향 없음**: opportunity·Oracle−CE·pairwise 라벨·specialist·gate 전부 `arm_d` F1 / `U_task_raw_f1`(raw PIU) 기반. control(`arm_ctrl`)은 진단축(raw-vs-ctrl 비교, Gate 0/1 ⑤)에서만 사용.
  - 논문 문구 수정 필요: "Controls must not have been previously consumed" → 삭제/정정. 이미 소비된 control은 사실상 중복 주입(≈no-op)이라 보수적 방향.
- **CE baseline 정합성**: pool-level CE argmax는 항상 D_t에 포함 (5세팅 3,881 states 예외 0) → 본문 argmax_{D_t}와 supp의 argmax_{C_t} 동치.
- **Case study 후보 (lockbox, 5ae5d06a…::s1)**: 재현 replay 미포함 상태(20% 감사 밖) — 단일 greedy replay 사례임을 명시할 것.
