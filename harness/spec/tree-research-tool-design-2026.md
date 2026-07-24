# 트리형 후속-연구-체인 탐색 도구 설계 (deep_tree_research)

> 답변 기반으로 후속 질문을 트리(DAG)로 확장하는 신규 도구. 병렬 리서치 2건(선행연구 비교 / 엔지니어링 설계) 수렴 결과.
> Tier B 성격이지만 통합면(fork + gptr-mcp)이 Tier A와 동일 → 하네스 6번째 스테이지로 편입.

## 결론 (한 줄)
**MindSearch식 명시적 트리(DAG) + Self-Ask 답변→자식 확장 + best-first frontier(novelty·gap 스코어, breadth÷2 funnel 폐기) + topological 병렬 leaf 리서처 + 계층적 bottom-up 종합.** MCTS는 배제(부분상태 가치평가 필요 → 연구엔 부적합, 비용만↑).

## 현재 deep_research.py와의 핵심 차이
| | 현재 (dzhng funnel) | 신규 (tree) |
|--|--|--|
| 확장 단위 | research goal의 후속질문 | **답변(answer) + gap** |
| 구조 | 암묵적 재귀, flat learnings | **1급 persisted 노드 트리(JSON)** |
| breadth | 기계적 ÷2 | **적응형**(gap 수·uncertainty 기반, 0도 가능=leaf) |
| 스케줄 | depth 소진 | **best-first 우선순위 큐** |
| 종료 | depth-only | 노드 resolved 신호 / novelty 임계 / 예산 / 수렴 |
| dedup | 없음 | 공유 visited-URL + 질문 임베딩 유사도 + boundary 프롬프트 |
| 종합 | flat concat 1콜 | **post-order 계층 roll-up + 전체 트리 citation map** |

## 선행연구 매핑 (근거)
- **MindSearch**(arXiv:2407.20183): WebSearchGraph(node={content,type}, adjacency_list), 코드젠으로 그래프 점증 구축, topological 병렬 실행, `response` 노드=LLM 판정 수렴. 답변이 planner로 되먹여져 다음 확장 구동. → **데이터 모델·병렬 실행의 레퍼런스.**
- **Co-STORM**(arXiv:2408.15232): moderator가 "retriever가 찾았지만 안 쓴 정보"에서 질문 생성 = **gap/novelty 확장**. mind map=개념 트리, reorganize()→계층 종합. → **gap-driven frontier·계층 종합의 레퍼런스.**
- **Self-Ask**(ofir.io/self-ask.pdf): 답변을 되먹여 다음 follow-up 생성 = **answer→child 순수 원형.**
- **Novelty-ToT**(arXiv:2605.06040): yes/no novelty prune = **크로스브랜치 dedup + frontier 스코어.** BFS가 DFS보다 토큰 효율↑.
- **MCTS 서베이**(arXiv:2510.09988): 연구(=발견 union)엔 best-first+novelty/gap이 지배, MCTS는 단일 최적경로용.

## 노드 스키마 (핵심)
```python
@dataclass
class ResearchNode:
    id: str                 # path-encoded "0.2.1" (sortable)
    question: str
    parent_id: str | None
    depth: int
    status: NodeStatus      # pending|researching|answered|expanded|pruned|failed
    answer_md: str          # 전체 인용 답변 (최종 리포트용)
    answer_digest: str      # ≤120단어 (자식·부모에 주입되는 유일 필드)
    learnings: list[str]    # 원자적 주장, 각자 source id
    sources: list[Source]
    novelty: float          # 조상·형제 대비 신규 learnings 비율
    priority: float         # frontier key
    gap_flags: list[str]    # unquantified_claim | no_recent_source | conflicting_sources
    uncertainty: float
    tokens_spent: int; credits_spent: float
    children: list[str]; question_embedding: list[float] | None
```
- `answer_digest`만 위/아래로 흐름 → lead context가 트리 크기와 무관하게 평평 (Anthropic의 "요약만 전달" = 25k 워드캡 문제를 구조적으로 해결).

## frontier 우선순위
```
priority = .40*novelty + .25*(gap/3) + .20*uncertainty + .15*parent.priority − .10*depth
```
자식 질문 생성 = 부모 **answer+gap** 조건 (원 쿼리 아님), 형제 질문 목록을 프롬프트에 주입해 intra-parent 중복 방지. 자식 수 = `min(max_breadth, len(gap_flags)+ceil(uncertainty*max_breadth))`, depth로 스로틀.

## dedup 3계층 (싼 것부터)
1. 트리 전역 **visited-URL set** → 재스크랩 크레딧 0 (fork가 이미 threading).
2. 질문 **임베딩 코사인** ≥0.92 drop / 0.83–0.92 near_dup 태그+novelty 감점 (임베딩은 이미 OpenAI).
3. 형제 목록 **boundary 프롬프트** ("각 질문은 disjoint, 겹치면 merge").

## 종료 기준 (기본값)
max_depth=3 · max_nodes=40 · max_breadth=4(적응↓) · token_budget=300k(~15×) · per_branch=80k · credit_budget=150 · novelty_prune<0.30 · 연속2회 저novelty→수렴 · wall=600s. 예산은 frontier pop 전 체크, 종합용 15% reserve. 예산 소진 시 잔여 PENDING="unexplored frontier"로 보고(트리는 여전히 종합 가능).

## 병렬 / 종합
- **wave = max(형제) not sum**: top-K frontier 동시 실행, wave간 순차(자식이 다음 wave dedup에 반영). backend=async GPTResearcher(기본 concurrency 2–3) | Claude SDK subagents(격리 context, final message만 반환, scale-up).
- **워커는 워커를 안 낳음** → Anthropic 10× 재귀 blowup 구조적 차단.
- **종합**: post-order walk, leaf digest→부모 roll-up→root. 전체 트리 citation map(URL union, 안정 [n] id).

## 통합 (권장)
**(b) fork에 신규 `gpt_researcher/skills/tree_research.py: TreeResearchSkill` + gptr-mcp에 `deep_tree_research` 툴.** deep_research.py 인플레이스 확장은 회귀 위험(선형 재귀+flat mutating). Python 오케스트레이터가 결정적 부분(frontier·예산·dedup·종합) 소유, LLM은 노드 리서치+자식질문만. 컨테이너(port 8123)에서 그대로 구동.

```python
async def deep_tree_research(
    query, max_depth=3, max_breadth=4, max_nodes=40,
    token_budget=300_000, credit_budget=150,
    expansion_policy="best_first",  # best_first|bfs|dfs
    novelty_threshold=0.30, max_concurrency=4,
    backend="async",                # async|subagents
    recency=None, stream=True,
) -> dict:  # {report_md, tree, citation_map, stats}
```

## 미해결 설계결정 1개
**strict tree vs DAG.** Agent1=DAG(크로스링크=dedup, MindSearch식) / Agent2=strict tree(dedup는 URL set+임베딩 레이어). **권장: strict tree JSON + dedup 레이어** (post-order 종합이 깨끗, 90% 커버) — 크로스링크는 dedup이 실측으로 부족할 때만 승격. `ponytail:` tree 먼저, DAG는 필요 입증 시.

## 시각화
`on_progress`/websocket 재사용, 노드 상태변화마다 JSON activity 이벤트(node_created/answered/expanded/pruned, wave_start/end, budget_warning, done). 산출: tree.json(스트리밍 패치) + mermaid + ASCII outline.

## 출처
MindSearch arXiv:2407.20183 / github.com/InternLM/MindSearch(Appendix G) · Co-STORM arXiv:2408.15232 / stanford-oval/storm · Self-Ask ofir.io/self-ask.pdf · Novelty-ToT arXiv:2605.06040 · MCTS 서베이 arXiv:2510.09988 · Anthropic multi-agent(15×, 요약전달) · Firecrawl /v2/search(categories,tbs) · Claude Agent SDK subagents · dzhng/deep-research
