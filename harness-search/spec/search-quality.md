# search-quality — deep_tree_research 탐색 파이프라인 품질 스펙

목표: `deep_tree_research`(gpt-researcher fork + gptr-mcp)를 firecrawl 멀티에이전트 방식 대비
**6개 지표(S1~S6) 전부**에서 우위로 만든다. 이 문서가 모든 스테이지의 완료조건(review 에이전트가
받는 유일한 판단 기준)이자 evidence 계약의 원본이다.

- Repos: `D:/dev_ext/gpt-researcher` + `D:/dev_ext/gptr-mcp`, 둘 다 branch `feature/search-quality`.
- 실행 대상: 컨테이너 `gptr-mcp-server` (port 8123, streamable-http, `/mcp`).
  `server.py`·`gpt_researcher/`는 컨테이너에 **마운트**되어 있다 → 코드 변경 후
  `docker compose -f D:/docker/gptr-mcp/docker-compose.yml up -d --force-recreate gptr-mcp`
  (measure/benchmark 스크립트가 자동 수행) 후 `/health` 200 확인 없이는 어떤 live 측정도 무효.
- 스택: Python 3.11 / pytest / ruff. 테스트 python = `../venv/Scripts/python` (harness-search/ 기준).
- 테스트 위치: `../tests/search_quality/s{1..5}/`.
- 커밋: `feature/search-quality` 에만, 메시지 prefix `[sq][s{N}]` (s0는 `[sq][s0]`).

## 실측된 결함 6층 (추정 아님 — 컨테이너 로그·tree.json·리포트로 관측)

1. **Retriever 전면 고장**: tavily 432 전량 실패, wikipedia 언어코드가 `wt`로 잘못 파싱되어
   DNS 실패, 그 결과 다수 노드에서 `Scraping 0 URLs`.
2. **컨텍스트 기아**: 노드당 research context median 1.3~8KB (필요 수준 대비 1/10 이하).
   모든 하류 결함의 근원.
3. **빈손 노드가 ANSWERED**: 스크랩 0건/컨텍스트 미달 노드가 LLM 사전지식으로 답을 만들어
   `NodeStatus.ANSWERED`로 트리에 남는다 → 오탐(traps 히트)의 직접 원인.
   (`gpt_researcher/skills/tree_research.py` — FAILED 전이는 예외 시에만 존재.)
4. **확장질문이 커버리지 무관**: `generate_child_questions`가 골든 영역(미커버 주제)을 향하지
   않고 표면적 파생 질문을 만든다.
5. **프루닝 무력**: `compute_novelty`가 문자열 매칭 기반 → 관측된 모든 tree.json에서
   `pruned_count: 0` (bun-rust-port 실측: node_count 13, pruned_count 0).
6. **인용·취합 무결성 붕괴**: `node.sources`가 "실제 읽고 인용한 문서"가 아니라 "retriever가
   반환한 URL"이며(bun 실측: 13노드에 citations 4개), rollup이 노드 답변과 모순되거나
   노드 근거 없는 주장을 만든다. `CitationAgent`(`gpt_researcher/skills/citation_verification.py`,
   커밋 3f0d0f8b)는 deep_research 경로에만 연결되어 있고 tree 경로에는 없다.

## 지표 정의 (S1~S6) — 전부 결정적, LLM 호출 금지

채점 대상 = 최종 리포트 `.md` + `tree.json` (deep_tree_research 산출) + 골든셋 1건.
점수는 0..1 float, evidence 에는 `round(100*x)` 정수 pct 로 기록한다.

| 지표 | 정의 | 방향 |
|------|------|------|
| S1 인용 무결성 | 리포트의 인용 `[id]` 중, citations 맵의 URL을 재fetch(캐시)한 본문에 앵커 문자열(인용 주변 핵심 구절, 정규화 부분매칭)이 실제 존재하는 비율. citations 맵에 없는 `[id]`(근거 없는 인용)는 분모에 포함하고 실패로 센다. | ↑ |
| S2 사실 재현율 | 골든 `facts[].pattern`(정규식) 중 리포트에 매칭된 비율 | ↑ |
| S3 오탐율 | 골든 `traps[].pattern`(must_not_match) 중 리포트에 매칭된 비율 | ↓ |
| S4 1차 출처 커버리지 | 골든 `required_primary_domains[]` 중 citations/sources URL 도메인에 존재하는 비율 | ↑ |
| S5 영역 커버리지 | 골든 `coverage_areas[].pattern` 중 리포트가 다룬 비율 | ↑ |
| S6 취합 일관성 | contested 병기율 × (1 - 모순·무근거 페널티): 골든 `contested[]` 각 항목은 `values`의 정규식 **2개 이상**이 함께 리포트에 등장해야 인정. 리포트의 수치 주장 문장 중 어떤 노드 답변과도 (정규화 토큰 매칭으로) 대응되지 않는 것은 `unsupported_claims`, 노드 답변과 수치가 상충하는 것은 `contradictions`로 세고 페널티. | ↑ |

## 골든셋 스키마 (`bench/golden/*.json`, 5개 이상)

```json
{
  "id": "bun-rust-port",
  "query": "<deep_tree_research에 넣을 질문 전문>",
  "category": "recent-event",
  "measure_pair": true,
  "required_primary_domains": ["bun.com", "github.com"],
  "facts": [{ "pattern": "<regex>", "desc": "왜 사실인지 + 출처" }],
  "traps": [{ "pattern": "<regex>", "desc": "왜 오탐인지" }],
  "contested": [{ "topic": "...", "values": ["<regexA>", "<regexB>"] }],
  "coverage_areas": [{ "id": "harness-design", "pattern": "<regex>" }]
}
```

### 다양성 하한 (s0 게이트 `diversity_ok`가 기계 강제)

벤치마크가 "유의미하게 다양한 케이스"를 포함해야 6개 지표가 전부 실제로 변별된다.
시대 불변(best-practice류) 쿼리만으로 채우면 빈손 노드가 사전지식으로 정답을 맞혀
**S3(traps)가 공허하게 통과**하고, 출처가 아무 블로그나 되는 쿼리는 **S4가 무의미**해진다.

- **골든당 최소치**: `facts >= 5`, `traps >= 3`, `coverage_areas >= 4`,
  `required_primary_domains >= 2`, `contested >= 1`, `category` 필수.
- **category 분류** (택1): `recent-event`(최근 사건·엔지니어링 사례) /
  `technical-deep-dive`(심층 기술) / `market-landscape`(시장·벤더 지형) /
  `contested-forecast`(전망이 갈리는 산업 주제) / `academic`(논문 레인 유도) /
  `encyclopedic-entity`(위키피디아 레인 유도 — s1의 언어코드 버그 경로를 live로 밟는다).
- **셋 전체**: 서로 다른 category **4종 이상**; **dated 골든 2개 이상**(fact의
  pattern/desc가 `202[4-9]`에 매칭 — 사전지식-적대성의 결정적 프록시);
  `measure_pair` 2개는 **category가 서로 다르고** 그중 1개 이상 dated;
  required_primary_domains 합집합 **6개 도메인 이상**(동일 도메인 클러스터 방지).
- **권고**(게이트 아님): 5개 중 1개는 academic 또는 encyclopedic-entity로 두어
  SmartRetriever의 비-general 레인을 최소 한 번 live로 태울 것. 기존 outputs/ 후보 중
  "transactional outbox 실패 모드"와 "denormalized derived table 설계"는 둘 다
  timeless technical-deep-dive라 **동시 채택 금지**(하나를 dated/encyclopedic 쿼리로
  교체하거나 contested·coverage 부담을 지울 것).

- `id: "bun-rust-port"` 골든셋은 **필수**이며, 이전 세션 확정 사실로 시딩한다. 근거 산출물:
  `D:/dev_ext/gptr-mcp/outputs/how-did-bun-port-530000-lines-of-zig-to-7c495f26.tree.json`
  (+ 같은 이름 `.tree-report.md`) — query 전문은 그 `meta.query`를 그대로 쓴다.
  확정 사실(예): Zig→Rust 530,000라인/11일/Claude Code 에이전트 하네스, git worktree 샤딩,
  PORTING.md 룰 파일, LIFETIMES.tsv, implementer/adversarial-reviewer 분리, 16,000 컴파일
  에러 워크큐, 테스트+보안리뷰+퍼징 검증 계층. traps 는 그럴듯한 오답(예: 다른 언어쌍,
  자릿수 다른 규모, "수작업 포팅")으로 구성. 각 fact/trap 은 1차 출처(bun.com 블로그,
  github)로 검증한 뒤 desc 에 출처를 적는다.
- 나머지 4개는 s0 에이전트가 자율 선정한다. 우선 후보(이미 tree 실행 이력이 있는 쿼리,
  `D:/dev_ext/gptr-mcp/outputs/` 참조): distributed-systems failure modes / denormalized DB
  design / edge-AI face recognition access control / solid-state battery supply chain.
  선정 기준: 검증 가능한 1차 출처 존재, trap 함정 구성 가능, contested(복수 관점 수치) 존재.
- `measure_pair: true` 는 정확히 2개(그중 하나는 bun-rust-port). s2~s5 measure 는 이 2개만
  live 실행하고, s6-benchmark 는 5개 전부 실행한다.

## 채점기 계약 (`bench/score_report.py`) — s0 산출물

```
python bench/score_report.py --golden bench/golden/<id>.json \
    --report <report.md> --tree <tree.json> --out <scores.json> \
    [--fetch-cache no_read/fetch_cache]
```

- **결정적. LLM 호출 금지** (`llm_calls` 필드는 항상 0; LLM SDK import/호출이 소스에 있으면
  s0 게이트가 즉시 FAIL). 유일한 네트워크 = S1 인용 URL 재fetch이며 `--fetch-cache` 디렉터리에
  URL-hash 키로 캐시해 재채점을 멱등으로 만든다.
- 출력 scores.json 필수 필드:
  `{"qid", "S1".."S6": float, "S1_pct".."S6_pct": int, "citations_total", "citations_grounded",
  "uncited_ids", "facts_matched", "facts_total", "traps_hit", "domains_hit", "areas_hit",
  "contested_ok", "contradictions", "unsupported_claims", "llm_calls": 0}`
- 자기검증 픽스처: `bench/fixtures/{good,bad}_report.md` + `{good,bad}_tree.json`
  (지정 골든 1건 기준). good 은 S1,S2,S4,S5,S6 **전부에서 bad 보다 엄격히 높고** S3 는 엄격히
  낮아야 한다 — 이것이 "채점기의 []가 파싱실패가 아님"의 증명이다 (law 5).

## 베이스라인 (`bench/baseline_firecrawl.json`) — s0에서 1회 실측 후 동결

- s0 에이전트가 골든 쿼리 5개 각각에 대해 firecrawl 멀티에이전트 방식(병렬 Claude 리서치 +
  firecrawl 검색; FIRECRAWL_API_KEY 는 `D:/docker/gptr-mcp/.env`)으로 리포트를 작성해
  `bench/baseline_runs/<id>.md`(+ 사용 소스 목록 `<id>.sources.json`)로 저장하고, **같은
  score_report.py**로 채점해 고정한다. 이때 tree.json 이 없으므로 S6 의 노드 대응 검사는
  sources.json 기반으로 동일 규칙 적용(채점기가 --tree 없이도 동작해야 함).
- 형식: `{"queries": {"<id>": {"S1_pct"..."S6_pct", "report": "...", "scores": "..."}},
  "aggregate": {"S1_pct"..."S6_pct"}}` (aggregate = 산술평균, 정수 반올림).
- s0 게이트 통과 시점 이후 `bench/golden/*`, `bench/baseline_firecrawl.json`,
  `bench/score_report.py`(채점기 본체 — baseline 이 이 바이트로 산출됐다) 는 **읽기 전용**.
  `bench/manifest.sha256`(freeze_bench.py 산출)과의 해시 대조를 이후 모든 게이트가 수행하고,
  harness-audit 이 git log 로 빌드 중 수정 여부를 재확인한다. 수정 발견 = 즉시 FAIL.

## 스테이지별 완료조건 (review 에이전트에 전달되는 기준)

### s1 — retriever 복구 (결함 1·2)
- tavily 432 실패 경로가 재시도/대체 라우팅으로 처리되고, wikipedia 언어코드 버그(`wt`)가
  수정되며, 스크랩 0건 상태가 조용히 지나가지 않는다.
- 단위테스트(mock, 결정적): 언어코드 정규화, 432 응답 시 대체 retriever 사용, 스크랩 0건 시
  에러 카운트 증가.
- live measure (골든 5쿼리, 컨테이너 내 `conduct_research` probe):
  `scraped_pages_per_query >= 3`(최소값 기준), `retriever_errors == 0`,
  `context_chars_median >= 20000`.

### s2 — 인용 무결성 (결함 6a)
- `CitationAgent` 를 tree 경로(`tree_research.py`)에 연결. `node.sources` 를 "retriever 반환
  URL"이 아니라 "실제로 읽고 인용한 문서"로 좁힌다. 리포트의 모든 `[id]` 는 citations 맵에
  존재해야 한다(fail-closed: 근거 없는 `[id]` 1개라도 있으면 실패).
- 단위테스트: 인용 좁히기, 미인용 소스 제외, uncited id 검출.
- live measure (measure_pair 2쿼리): `S1_pct >= 80` (전 쿼리 최소값), `uncited_ids_total == 0`.

### s3 — 빈 컨텍스트 fail-closed (결함 3)
- 스크랩 0건 또는 컨텍스트 임계 미달 노드는 `ANSWERED` 가 아니라 `FAILED` 로 전이한다.
  FAILED 노드는 rollup/리포트의 근거로 쓰이지 않는다.
- 단위테스트(필수, 에이전트 자기보고 금지): 빈 컨텍스트 주입 시 노드 status 가 FAILED,
  FAILED 노드 답변이 synthesis 에 미포함.
- live measure (measure_pair 2쿼리): `traps_hit_total == 0`.

### s4 — 확장질문 커버리지 지향 + 임베딩 novelty (결함 4·5)
- `generate_child_questions` 가 "아직 커버되지 않은 영역"(부모까지의 트리가 다룬 주제와의
  차이)을 향하도록 개선. `compute_novelty` 를 문자열 매칭 → 임베딩 코사인 기반으로 교체.
- 단위테스트: 임베딩 novelty(mock 임베딩), 저novelty 자식이 PRUNED, 커버리지 지향 프롬프트
  구성.
- live measure (measure_pair 2쿼리): `S4_pct >= 85`(최소값), `pruned_count_total > 0`
  (프루닝이 실제 동작한다는 양의 증거), `s5_improved == 1` (측정 쿼리 평균 S5_pct 가
  baseline 평균 초과).

### s5 — 취합 모순 검사 (결함 6b)
- rollup 결과가 노드 답변과 모순되지 않는지 검사하는 패스 추가. 최종 리포트의 각 주장이
  최소 1개 노드 답변에 근거하는지 매칭; 무근거 주장은 제거 또는 노드 재조사.
- 단위테스트: 모순 검출, 무근거 주장 검출/제거.
- live measure (measure_pair 2쿼리): `contradictions_total == 0`, `unsupported_claims_total == 0`.

### s6-benchmark — 최종 대결
- 골든 5쿼리 전부 deep_tree_research 실행(라운드별 캐시), score_report 채점, baseline 과
  aggregate 비교. 게이트: S1,S2,S4,S5,S6 각각 baseline aggregate **초과** AND S3 는 baseline
  **이하**. `queries_scored >= 5` 미만이면 hollow zero 로 FAIL. 미달 시 가장 낮은(부족분 최대)
  지표의 담당 스테이지로 되돌아간다: S1→s2, S2→s1, S3→s3, S4→s4, S5→s4, S6→s5.
  재도전 한도 3라운드.

## Live evidence 계약

- s1 probe: `scripts/container_probe_s1.py` 를 `docker cp` 로 컨테이너에 넣고 실행,
  GPTResearcher 를 직접 introspect 해 한 줄 JSON
  (`{"query_id","scraped_pages","context_chars","retriever_errors"}`)을 출력한다.
  retriever_errors 는 retriever/scraper 로거의 ERROR 레코드 + 예외 카운트.
- s2~s6: MCP `deep_tree_research` 호출 → 반환된 호스트 경로에서 tree.json + report 수거 →
  `no_read/bench_runs/round{R}/<id>.*` 에 캐시(있으면 재실행 생략 = 중단 재개 가능) →
  score_report 채점.
- tree.json 은 기존 계약 유지: `meta.{query,budget_respected,max_depth_reached,pruned_count,
  node_count}`, `nodes[].{id,depth,status,question}`, `citations{id:url}`. s5 이후 노드 답변
  텍스트가 채점에 필요하므로 `nodes[].answer`(요약 답변 텍스트)를 **추가**한다(하위호환 추가만).

## 하드 룰 (모든 노드 공통)

- `bench/golden/*` 와 `bench/baseline_firecrawl.json` 은 s0 이후 읽기 전용. 수정 = 빌드 실패.
- 테스트 삭제/스킵/약화 금지. RED 테스트 파일은 impl 중 byte-불변(게이트가 sha256 재계산).
- `harness-search/scripts/*`, `no_read/evidence/*` 손 편집 금지 — evidence 는 전부 스크립트
  산출물이며 재실행이 곧 재생성(멱등)이다.
- 코드 변경 후 컨테이너 재시작 필요(measure/benchmark 스크립트가 force-recreate 수행).
- 모든 셸 명령에 timeout 을 건다(이 프로젝트는 장시간 무응답 이력이 있다).
- 커밋은 feature/search-quality 에만, prefix `[sq][s{N}]`.
