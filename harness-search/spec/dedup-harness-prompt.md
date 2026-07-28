# /gralph-harness 프롬프트 — 취합 중복 제거 (dedup harness)

아래 전문을 `/gralph-harness` 에 그대로 붙여 넣는다. 인터뷰는 이 문서로 끝났으므로
Phase 0~2 를 이걸로 채우고 Phase 3(compose/prove/run)으로 직행하게 되어 있다.

---

대상 프로젝트: D:\dev_ext\gpt-researcher (+ D:\dev_ext\gptr-mcp)
하네스 위치: D:\dev_ext\gpt-researcher\harness-search\  (기존 하네스 **재사용**, 프로파일만 신규 `dedup.yaml`)
브랜치: feature/search-quality (양 repo, 계속 사용)

인터뷰는 아래 답으로 이미 끝났다. Phase 0~2를 이 답으로 채우고 Phase 3로 직행하라.
빠진 결정만 AskUserQuestion으로 물어라.

## Phase 0 — 프레임

- 목적: `deep_tree_research` 최종 리포트의 **취합 중복을 제거**한다. 리포트는 노드 답변을
  이어붙인 것이 아니라 **주장 단위로 병합된 종합**이어야 한다. 사실은 하나도 잃지 않는다.
- 스택: Python 3.11 / pytest / ruff. 테스트 python = `../venv/Scripts/python`.
- 실행 대상 컨테이너 `gptr-mcp-server`(8123, streamable-http, /mcp). `server.py`·`gpt_researcher/`
  는 마운트 → 코드 변경 후 `docker compose -f D:/docker/gptr-mcp/docker-compose.yml up -d
  --force-recreate gptr-mcp` + `/health` 200 확인.
- **기존 하네스를 재사용한다**. 새로 만들지 말 것:
  - `bench/golden/*`(5개), `bench/baseline_firecrawl.json`, `bench/score_report.py` 는 **동결**
    (`bench/manifest.sha256`). `scripts/check_frozen.py` 가 모든 게이트에서 재검증한다. 수정 = 즉시 실패.
  - `scripts/rollup_scan.py`(중복 측정), `scripts/s7_dedup.lua`(게이트), `scripts/code_fp.py`(신선도),
    `scripts/loop_audit.py`(캡을 journal 에서 유도), `scripts/lib.lua`, `scripts/verify_impl.py`,
    `scripts/review_diff.py`, `scripts/check_commit.py` 를 그대로 쓴다.
  - 라운드 캡·승인은 `no_read/audit/grants.json` 에 사람이 기록하는 기존 방식 유지.
- greenfield 아님. graph_source 는 `"on-disk-static"`.
- 스펙: `spec/search-quality.md` 의 s7 절이 이미 완료조건이다. 부족하면 그 절을 확장하라.

## 실측 baseline (추정 아님 — round4 5쿼리 + verify1 1쿼리 실측)

| 지표 | round4(5쿼리) | verify1(outbox, 최신 코드) |
|---|---|---|
| lifted / kept | 7/7, 5/5, 5/5, 10/10, 8/8 — **전부 일치** | **12/12** |
| max_lift | 5쿼리 모두 100% | 100% |
| 합성비(리포트/kept 답변합) | 120~133% | **129%** |
| headings | 3~6 | **2** |
| 리포트 크기 | 33k~69k자 | **81k자** |
| S2(사실 재현) | aggregate 80 | outbox 100 |

즉 **kept 노드 답변이 100% 그대로 리포트에 실린다**. 병합이 0이다.

**핵심 함정 — 문장 단위 중복 스캔은 이 결함을 못 잡는다.** 같은 파일에서 반복 5-gram 0~2%,
섹션 쌍 최대 Jaccard 0.22 로 "중복 없음"이 나온다. 붙여넣은 답변은 각각 고유한 산문이고
형제 노드 간 겹침은 **주제 차원**이다. 어휘 유사도를 쫓지 마라. 측정 대상은 **붙여넣기 자체**다.

**상류 원인은 이미 조사됐다**(`no_read/audit/pending_rca.md`): 확장기가 PENDING 큐를 못 봐서
형제가 사실상 같은 질문을 조사한다. 상류 완화(queued-ground 노출)는 이미 적용됐고
**중복은 줄지 않았다**(verify1에서 리포트가 오히려 길어짐). 그러므로 이 하네스는
**취합 자체**를 고친다.

## Phase 1 — 스테이지 그래프

  s0-resynth → s1-red → s1-impl → s1-review → s1-offline → s2-live → harness-audit → DONE

에이전트는 노드로 분리한다(같은 세션 겸임 금지): `*-impl`(구현만), `*-review`(git diff +
완료조건만 받고 "이 코드는 틀렸다고 가정하고 왜 틀렸는지 대라", 코드 수정 금지),
`harness-audit`(게이트 자체 감사).

## Phase 2 — 스테이지별 게이트 조건

### s0-resynth — 오프라인 재취합 러너 (**나머지 전부가 여기 의존**)
왜 필요한가: 취합을 고칠 때마다 live 재실행하면 쿼리당 ~330 크레딧·12분이 든다.
`synthesize_node` 는 retrieval 호출이 전혀 없는 **LLM 전용**이고 리포트 조립도 `self.nodes`
만 쓰므로, 캐시된 tree 에서 **취합만 다시 돌릴 수 있다** → 크레딧 0, 사이클 수 분.

**단, 빠진 조각이 하나 있다**: 인용 부착이 `self._read_docs`(스크랩 원문)를 쓰는데 tree.json
에 없다. 따라서 s0 은 두 가지를 한다.
1. **아티팩트 계약 확장**: tree 산출 시 `<name>.read_docs.json`({url: text})을 함께 저장.
   기존 tree.json 스키마는 건드리지 말고 **추가만** 하라.
2. `scripts/resynth.py --tree <tree.json> --read-docs <read_docs.json> --out <dir>`:
   캐시된 노드 답변으로 `self.nodes` 를 복원해 **취합 경로만** 재실행하고 리포트를 쓴다.
   retrieval·임베딩 호출 금지(호출하면 게이트 실패).

게이트(`s0_resynth.lua`):
- **충실도 증명이 핵심이다.** 코드를 바꾸지 않은 상태에서 캐시 tree 를 재취합하면 원본과
  같은 중복 지표가 나와야 한다: `fidelity_lifted_delta == 0`, `abs(fidelity_ratio_delta) <= 5`,
  `fidelity_s2_delta >= -2`. 이게 깨지면 오프라인 루프는 허구를 최적화하게 된다(law 4/5).
- `resynth_retrieval_calls == 0` (소스 정적 스캔 + 런타임 카운터 양쪽)
- `read_docs_persisted == 1`, `read_docs_urls >= 10` (양의 결속 — 빈 read_docs 는 공허)
- 충실도 증명에 쓸 tree 는 **read_docs 를 포함한 fresh live 실행 1건**이 필요하다
  (기존 round0~4·verify1 캐시에는 read_docs 가 없다). 골든 1개, ~330 크레딧.
- `frozen_ok`, `[sq][d0]` 커밋

### s1 — 취합 병합 (본 작업)
완료조건:
- `synthesize_node`/리포트 조립이 **노드 답변을 붙여넣지 않는다**. 겹치는 발견은 주장 단위로
  하나의 진술로 병합하고, 조직화된 섹션(헤딩)으로 배치한다.
- 사실 손실 금지. 인용 id 체계는 깨지 말 것(S1 채점이 `[id]` 마커 위치에 의존한다 —
  `bench/score_report.py` 의 마커 정규식 `\[(\d{1,3})\](?!\()` 을 반드시 확인하고 맞춰라).
- 단위테스트(RED, mock LLM, 결정적): 같은 주장을 담은 두 노드 답변이 **한 번만** 리포트에
  나타난다 / 서로 다른 주장은 둘 다 살아남는다 / 인용 id 가 유지된다 / 노드 답변 원문이
  70% 이상 그대로 실리지 않는다.

`s1-offline` measure 게이트 (**크레djit 0, 골든 5개 전부 재취합**):
- `queries_scanned == 5`, `node_answers_scanned_total >= 20` (hollow zero 방지)
- `lifted_nodes_max <= 1` (baseline 12/12 → 사실상 0에 가까워야 한다)
- `synthesis_ratio_pct_max <= 70` (분모 = **kept 노드**; baseline 120~133%)
- `headings_min >= 4` (baseline 2~6)
- **`s2_aggregate_pct >= 80`** ← 안티치트. 중복 지표는 **내용을 지우면 가장 싸게 통과**한다.
  동결 채점기의 사실 재현율이 연결 버전 수준 아래로 떨어지면 실패.
- `code_fp` 신선도(게이트가 재계산), `frozen_ok`, `[sq][d1]` 커밋

### s2-live — 오프라인 결과가 실제로 성립하는지 1회 확인
- 골든 1개(**outbox-failure-modes**, 중복이 가장 심했던 케이스)를 실컨테이너로 실행하고
  같은 지표를 잰다. 오프라인과 어긋나면 오프라인 측정이 거짓말한 것이다.
- 게이트: `live_lifted_nodes <= 1`, `live_synthesis_ratio_pct <= 70`, `live_headings >= 4`,
  `live_S2_pct >= 88`(verify1 실측 100, round4 88 — 낮은 쪽 기준), `live_S3_pct <= 0`,
  `live_S1_pct >= 95`(verify1 99), `recreated=true`, `health=200`, `code_fp` 일치
- **오프라인↔live 정합**: `abs(live_ratio - offline_ratio_for_that_query) <= 10`
  (어긋나면 s0 의 충실도 증명이 틀렸다는 뜻 → s0 로 되돌린다)

### harness-audit — 게이트 감사 (기존 `scripts/audit_check.py` 재사용, 노드 목록만 갱신)
law 6 try 매트릭스 / law 8 3자 동기 / law 10 regen 멱등 / 점수 조작 git 검사 /
`store_untampered`(캡을 store 아닌 journal 에서 유도).

## 하드 룰 (모든 노드 guidance 에 넣을 것)

- `bench/golden/*`, `bench/baseline_firecrawl.json`, `bench/score_report.py` 는 **읽기 전용**.
  수정 시 빌드 실패(manifest + audit git 이중 검사).
- 테스트 삭제/스킵/약화 금지. `no_read/evidence/*` 손 편집 금지.
- **`.gralph/**` 는 프레임워크 소유** — cursor 되감기·카운터 리셋 금지. 캡은 append-only
  journal 에서 유도되므로 store 를 고쳐도 바뀌지 않고 조작 흔적만 남는다. 캡에 막히면
  멈추고 보고하라(과거에 에이전트가 캡을 스스로 리셋한 전례가 있다).
- 오프라인 러너를 **live 대용으로 신뢰하기 전에 반드시 s0 충실도 게이트를 통과**해야 한다.
- live 실행 스크립트는 **포그라운드로 실행하고 끝날 때까지 기다린다**. 백그라운드로 띄우고
  턴을 종료하면 프로세스가 고아가 되고 루프가 iteration 을 태운다(실측 8회 손실).
- 모든 Bash 호출에 timeout. 커밋은 `feature/search-quality` 에만, prefix `[sq][d{N}]`.

## 루프 정책

- fail_threshold: red/impl 3, review/measure/audit 2
- 모델: impl/red/s0/audit 기본 티어, review/measure sonnet
- agent timeout: 기본 75m, s2-live 180m. lua_timeout: 기본 90s, impl/offline 900s, audit 3600s
- `run-until-done.sh` (GALP launcher 금지). 세션 회전에 죽지 않게 `run-detached.cmd` 로 띄울 것
- 리뷰 캡 3라운드, 도달 시 **사람 판단** — grants.json 에 사유와 함께 기록해야 연장된다

---

## 설계상 짚어둘 것

**s0 가 실패하면 전부 무의미하다.** 오프라인 재취합이 live 와 다른 결과를 내면 나머지
스테이지는 가짜 지표를 쫓는다. 그래서 s0 게이트의 핵심은 러너가 존재하느냐가 아니라
**코드 무변경 재취합이 원본을 재현하느냐**(fidelity delta)다. 원래 하네스에서 채점기에
`llm_calls == 0` 을 요구했던 것과 같은 자리의 조건이다.

**안티치트가 반대 방향이라는 점에 주의하라.** 다른 게이트는 대개 "더 많이/더 잘"을
요구하지만 중복 게이트는 "더 적게"를 요구한다. 그래서 **내용을 지우는 것이 가장 싼 통과
경로**다. `s2_aggregate_pct >= 80` 이 그것을 막는 유일한 장치이므로 절대 완화하지 말 것.
