# /gralph-harness 빌드 프롬프트 — gpt-researcher Tier A 업그레이드

아래를 `/gralph-harness` 입력으로 사용. 인터뷰 스킬이므로 이 프롬프트는 프로젝트 substrate + 이미 확정한 인터뷰 답변을 선제공한다.

---

/gralph-harness

**프로젝트**: gpt-researcher fork(`/deep-research`)의 Tier A 품질 업그레이드를 fail-closed 자율 빌드 하네스로 구축한다. 신규 프로젝트가 아니라 **기존 포크에 6개 패치를 순차 적용**하는 하네스다 → Phase 0의 codegrove-project-stack 분기는 타지 말 것.

**대상 코드베이스**
- 소스: `D:\dev_ext\gpt-researcher` (ymanse fork, 0.15.1, 현재 브랜치 `upgrade/0.15.1`)
- 런타임: docker `gptr-mcp-server`, `D:\docker\gptr-mcp\docker-compose.yml`, host port **8123** → `/mcp` (streamable-http). health: `curl http://127.0.0.1:8123/health`
- **바인드마운트**: 컨테이너가 `D:\dev_ext\gpt-researcher\gpt_researcher` 를 mount → 소스 수정은 즉시 반영, **실행 반영은 recreate 필요**: `docker compose -f D:/docker/gptr-mcp/docker-compose.yml up -d --force-recreate gptr-mcp`
- LLM: `claude_agent:sonnet` (구독 CLI OAuth, API 비용 0), 임베딩 OpenAI. Firecrawl: `.env`에 `FIRECRAWL_API_KEY` 있음, research MCP 사용 가능.

**확정된 하네스 파라미터 (인터뷰 선답)**
- **적용/격리**: `D:\dev_ext\gpt-researcher` 에 새 feature 브랜치(`feature/tier-a-upgrade`) 생성, 스테이지별 게이트 통과 시 커밋. **git worktree 격리는 금지**(컨테이너가 그 경로를 mount 안 함 → 라이브 검증 깨짐). 브랜치 커밋은 바인드마운트 경로에서 그대로.
- **게이트 증거**: 모든 스테이지 **pytest(RED→GREEN) + 라이브 컨테이너 스모크** 둘 다. 게이트는 tool이 낸 evidence 파일을 Lua로 기계 검증(에이전트 self-report 금지).
- **범위**: 6 스테이지 전부 순차.
- **Firecrawl**: 키 있음, 라이브 검증이 firecrawl 크레딧 소모해도 OK (스테이지당 스모크 1~2콜로 절제).

**공통 게이트 규약 (모든 스테이지)**
1. 스테이지는 command-graph 노드. `PENDING→RED(테스트 작성/실패 확인)→GREEN(구현)→SMOKE(라이브)→COMMIT` 로만 전진.
2. **RED 게이트**: 새 pytest가 존재하고 구현 전엔 **실패**해야 함(evidence: pytest junit xml에 해당 테스트 FAIL). fail-closed: RED 없이 GREEN 금지.
3. **GREEN 게이트**: `pytest` 해당 테스트 전부 pass (evidence: junit xml exit 0). 기존 테스트 회귀 없음.
4. **SMOKE 게이트**: 컨테이너 `--force-recreate` 후 `curl /health` 200 + 아래 스테이지별 실제 MCP 호출의 어서션을 만족하는 **evidence JSON** 산출. Lua 게이트가 그 JSON의 필드를 검증(값 존재+임계 비교). 스모크 스크립트는 `no_read/evidence/stageN.json` 에 결과를 떨군다.
5. **COMMIT 게이트**: `[tier-a][stageN] ...` 커밋 존재. 게이트 4개 evidence 파일 모두 present+pass일 때만.
6. ralph 루프: 게이트 실패 시 재시도. 예산: 스테이지당 실패 재시도 최대 3회, 초과 시 halt+사유.

---

## 스테이지 정의 (6)

### Stage 1 — SmartRetriever 라우팅에 firecrawl `/search` 추가 (최대 레버리지)
- **변경**: firecrawl retriever 어댑터 신설(`retrievers/firecrawl/`) — `/v2/search` (`categories`, `tbs`, `scrapeOptions.formats=[markdown]`) 호출, `{href,title,body}` 정규화(body=clean markdown). `smart_retriever.py` ROUTING_TABLE의 `general_web`·`comprehensive`·`news_current`에 `("firecrawl", n, {...})` 추가. `_RETRIEVER_API_KEYS["firecrawl"]="FIRECRAWL_API_KEY"`.
- **RED/GREEN pytest**: (a) firecrawl 어댑터가 mock API 응답을 `{href,body,title}`로 정규화, body 비어있지 않음. (b) SmartRetriever가 general_web 쿼리 라우팅에 firecrawl 포함. (c) 키 없으면 스킵(기존 availability 로직 회귀 없음).
- **SMOKE**: recreate 후 일반 쿼리로 `quick_search`(또는 deep_research) 호출 → evidence JSON: `{sources:[{url,body_len,via}]}`. **게이트 어서션**: firecrawl 경유 소스 ≥1 AND 그 body_len > 스니펫 임계(예 400자) = clean markdown 확인.

### Stage 2 — learnings 손실 압축 완화
- **변경**: `deep_research.py process_research_results`의 `num_learnings=3→8`, `max_tokens=1000→2500` (config화: `DEEP_RESEARCH_LEARNINGS`, `DEEP_RESEARCH_LEARNINGS_TOKENS`).
- **RED/GREEN pytest**: mock LLM이 10개 learning 반환 시 함수가 ≤8개 반환(기존 3 상한 아님), 인용 파싱 회귀 없음.
- **SMOKE**: 고정 쿼리로 deep_research → evidence JSON `{learnings_count}`. **어서션**: learnings_count ≥6 (baseline 3 대비 증가).

### Stage 3 — citation-verification 패스
- **변경**: 경량 CitationAgent — 핵심 learning의 인용 URL을 firecrawl `scrape(max_age=0)` 재취득 → 인용 quote 문자열/의미 매칭, 미매칭은 `unverified` 플래그. (또는 기존 `multi_llm_reviewer` 활성 경로와 결합.)
- **RED/GREEN pytest**: quote가 소스에 존재→verified, 없음→flagged. mock scrape로 결정적.
- **SMOKE**: 쿼리 실행 → evidence JSON `{total_claims, grounded, unverified}`. **어서션**: 모든 claim이 resolvable source 보유 AND grounded/total ≥ 0.7.

### Stage 4 — 논문 레인을 Firecrawl research MCP로 승격
- **변경**: SmartRetriever `academic` 라우팅을 arxiv+semantic_scholar 단독 → firecrawl research MCP(`research_search_papers` + `research_related_papers` mode=`citers`)로. `from`/`to` 최신성 필터, citers 확장으로 시드 인용 최신 논문 추가.
- **RED/GREEN pytest**: mock research MCP → 어댑터가 papers(id,title,date) 반환; citers 확장이 시드보다 새 논문 추가.
- **SMOKE**: 학술 쿼리(recency window 지정) → evidence JSON `{papers:[{id,date}], from_citers_count}`. **어서션**: papers ≥3 AND date가 window 내 ≥1 AND from_citers_count ≥1.

### Stage 5 — scope/clarification 옵션화
- **변경**: `deep_research.py run()`의 "Automatically proceeding" 자동답변을, MCP 파라미터 `scope: bool`로 게이트. on이면 1-round 브리프(원 쿼리→명확화 질문→스코프 확정) 생성 후 진행, off면 현행 유지(회귀 0).
- **RED/GREEN pytest**: scope=off → 자동진행(현행 동일, 회귀 없음); scope=on → 브리프 객체 생성.
- **SMOKE**: scope=on으로 호출 → evidence JSON `{brief_present:true, brief_len}`. **어서션**: brief_present=true AND brief_len>0. off 호출도 1회 → 현행 동작 유지 확인.

### Stage 6 — `deep_tree_research` 신규 트리 도구 (설계: `tree-research-tool-design-2026.md`)
- **변경**: `gpt_researcher/skills/tree_research.py: TreeResearchSkill` 신설 + gptr-mcp에 `deep_tree_research` 툴. MindSearch식 persisted 트리(JSON) + best-first frontier(novelty·gap) + Self-Ask answer→child + 공유 visited-URL·임베딩 dedup + post-order 계층 종합. 노드 리서치는 기존 `GPTResearcher` 재사용(backend=async 기본). 파라미터: `max_depth,max_breadth,max_nodes,token_budget,credit_budget,novelty_threshold,expansion_policy,stream`.
- **RED/GREEN pytest (결정적, 노드 리서치 mock)**: (a) best-first frontier가 priority 내림차순 pop. (b) novelty<0.30 자식 PRUNED, 미확장. (c) 질문 임베딩 코사인 ≥0.92 → dedup drop. (d) post-order 종합이 leaf→root 순 roll-up, citation map이 URL union·안정 id. (e) 예산 소진 시 잔여 PENDING 보고, 종합 여전히 산출.
- **SMOKE**: recreate 후 `deep_tree_research(query, max_depth=2, max_nodes=8, credit_budget=낮게)` → evidence: `tree.json` + `{report_md_present, max_depth_reached, node_count, pruned_count, citations_resolve, budget_respected}`. **어서션**: report_md_present=true AND max_depth_reached≥2 AND node_count≥3 AND citations_resolve=true AND budget_respected=true.

---

**하네스 산출물**: gralph 엔진(ralph 루프 + command graph + Lua 게이트) 구성 파일, 스테이지별 pytest + 스모크 스크립트 + Lua 게이트 스크립트, evidence 디렉터리(`no_read/evidence/`), 그리고 6 스테이지를 순차 수렴시키는 실행 엔트리. 각 게이트는 **evidence 파일의 기계 검증**으로만 전진하며, 삭제/스킵으로 우회 불가(fail-closed).

**인터뷰에서 내가 추가로 물어봐야 할 것** (gralph-harness가 판단): evidence 디렉터리 위치·Lua 게이트 임계값 미세조정·스모크 쿼리 고정 문자열·pytest 실행 위치(호스트 venv vs 컨테이너 exec)·재시도 상한. 위 substrate로 채워지지 않는 부분만 질문.
