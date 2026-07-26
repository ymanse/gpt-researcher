# HARNESS — search-quality build loop

gralph 하네스: `deep_tree_research`를 frozen firecrawl 베이스라인 대비 S1~S6 전 지표 우위로
끌어올린다. 스펙: [`spec/search-quality.md`](spec/search-quality.md), 인덱스: [`PRD.md`](PRD.md).

```
s0-bench → (s1..s5: red → impl → review → measure) → s6-benchmark → harness-audit → DONE
                         ↑____review blocking____|        |
              ↑______________s6 weakest-metric refit______|
```

실행: `./run-until-done.sh` (GALP launcher 금지 — Windows에서 구조적으로 깨짐).
상태: `.gralph/search-quality/` (state.json cursor / store.json / failures.json / journal.jsonl).

## 각 게이트가 증명하는 것

| 게이트 | 증명 | 결정적 근거 |
|---|---|---|
| **s0-bench** | 채점 인프라가 진짜다: golden_count>=5 + 스키마/정규식 컴파일(golden_schema_ok), bun-rust-port 시딩(bun_first), measure_pairs=2, **diversity_ok=1**(다양성 하한 — 골든당 facts>=5·traps>=3·areas>=4·domains>=2·contested>=1·category, 셋 전체 category 4종+·dated 골든 2+·measure_pair category 상이+dated 1+·도메인 합집합 6+; timeless 쿼리만으로는 S3가 공허 통과하므로), **llm_calls=0**(채점기에 LLM SDK 토큰 0 — law 2), **fixtures_passed=2**(good이 bad를 S1,S2,S4,S5,S6 전부에서 엄격히 이기고 S3는 엄격히 낮음 — law 5, []가 파싱실패가 아님의 증명), baseline_queries=golden_count(전 골든에 S1_pct..S6_pct + report/scores 출처), freeze(manifest) + `[sq][s0]` 커밋 | in-gate `bench_selfcheck.py` + `check_frozen.py` + `check_commit.py` |
| **sN-red** | RED 무결성(law 3): errors=0, collected>=1, passed=0, failed>=1, test_files sha256 + base_sha 기록, bench frozen_ok | `pytest_evidence.py`가 pytest junit에서 생성; in-gate `check_frozen.py` |
| **sN-impl** | GREEN 안티탬퍼: RED 파일 sha256 재계산 일치(hash_match), collected 비감소, 스테이지+누적 전체 suite green(failed/errors/skipped=0), ruff_errors=0(E9,F — 변경 파일 한정), frozen_ok, review_addressed_ok(직전 review의 blocking id 전부가 impl_ack의 addressed_findings에 존재 **AND** impl_ack의 review_head_sha가 그 review.json의 head_sha와 일치 — id는 라운드마다 재사용되므로 이 바인딩 없이는 이전 라운드 ack가 새 findings를 무효 충족) | **in-gate `verify_impl.py` 재실행** — 위조 evidence는 생존 불가 |
| **sN-review** | 분리 레인 적대 리뷰: 입력은 `review_diff.py`의 diff + 스펙 완료조건뿐. head_sha가 현재 HEAD와 일치(stale 리뷰 재활용 차단, in-gate 재계산), findings 배열 + blocking_count 숫자. blocking_count>0 → `sN-impl`로 라우트(store `rev:sN` 카운트, **3라운드 초과 시 hard fail**), 0 → `sN-measure` | 판정 자체는 LLM(의도된 law 2 예외, 아래 잔존 리스크) — 신선도·형식·라운드 캡은 결정적 |
| **sN-measure** | 실컨테이너 측정: recreated:true + health:200(bind-mount 반영), bench_round가 store와 일치(stale evidence 차단), **code_fp가 현재 구현 바이트와 일치**(게이트가 재계산 — 측정 후 구현을 고치면 그 evidence는 거부된다), 스테이지 임계값(아래), frozen_ok, `[sq][sN]` 커밋 + 양 repo 클린. refit 라운드면 s6-benchmark로 직행 라우트(bench_refit 클리어) | `measure.py`만이 evidence를 쓴다; 캐시 키가 (라운드, 골든, **코드 지문**)이라 멱등이면서도 코드가 바뀌면 live 재실행 |
| **s6-benchmark** | 최종 대결: queries_scored==golden_count>=5(미만은 hollow zero — law 4), S1,S2,S4,S5,S6 aggregate가 baseline **초과** AND S3 **이하** → all_pass=1 → audit. 미달 → weakest_metric(부족분 최대)의 담당 스테이지 impl로 라우트(S1→s2,S2→s1,S3→s3,S4→s4,S5→s4,S6→s5), bench_round 증가, **3라운드 캡** | `benchmark.py` + frozen 채점기 + frozen baseline |
| **harness-audit** | 게이트 자체 감사: try_ok(law 6 — 전 노드 known-good/bad `try_probe.py` 리포트 쌍), git_ok(**점수 조작 탐지** — `[sq][s0]` freeze 커밋 이후 bench/golden/*·baseline_firecrawl.json을 건드린 커밋 0), frozen_ok, regen_ok(law 10 — 재생성기 2회 실행 바이트 동일), sync_ok(law 8 — 게이트 토큰이 profile guidance와 이 문서에 존재) | in-gate `audit_check.py` |

측정 임계값 (실측 결함에서 도출):
- s1: queries_run>=5, scraped_pages_min>=3, retriever_errors_total=0, context_chars_median>=20000 (관측 median 1.3~8KB)
- s2: queries_run>=2, S1_min_pct>=80, uncited_ids_total=0 (근거 없는 [id] fail-closed)
- s3: queries_run>=2, traps_hit_total=0 (+빈 컨텍스트→FAILED 단위테스트는 s3 RED가 강제)
- s4: queries_run>=2, S4_min_pct>=85, pruned_count_total>=1 (관측: pruned_count 항상 0), s5_improved=1
- s5: queries_run>=2, contradictions_total=0, unsupported_claims_total=0

## 인터뷰 결정 사항 (이 하네스를 형성한 것)

- baseline은 **s0에서 1회 실측 후 동결**: s0 에이전트가 골든 5쿼리를 firecrawl 멀티에이전트
  방식으로 실행해 같은 채점기로 채점 → `bench/baseline_firecrawl.json` freeze. 이후 수정은
  manifest 해시 + git log 이중으로 적발되며 즉시 빌드 실패.
- **동결 대상은 골든셋 + baseline + `bench/score_report.py`(채점기 본체)** 세 가지다. 골든셋만
  잠그면 창문만 잠그고 문은 열어둔 꼴 — baseline 수치 자체가 **그 바이트의 채점기로** 산출됐기
  때문에, 채점기가 바뀌면 baseline 과의 비교가 성립하지 않는다. 채점기에 진짜 버그가 발견되면
  빌드는 하드스톱하고 사람이 판단한다(수정하려면 baseline 재측정이 동반돼야 한다).
- 골든셋 2~5는 s0 에이전트 자율 선정 (기존 `D:/dev_ext/gptr-mcp/outputs/` tree 실행 이력
  쿼리 우선). #1은 bun-rust-port 고정 시딩.
- impl/review/measure는 **세션 분리 노드** (같은 세션 겸임 금지): 구현자는 리뷰하지 않고,
  리뷰어는 diff+완료조건만 받고, 측정은 스크립트가 한다.
- s2~s5 measure는 measure_pair 2쿼리만 live 실행(런타임 통제), s1은 5쿼리 probe(가볍다),
  s6는 5쿼리 전부.
- ruff는 E9,F(correctness)만, 변경 파일 한정 — fork의 기존 스타일 부채(UP/BLE/I ~35건/파일)에
  게이트가 인질 잡히지 않게.

## 루프 정책

- stop 계층: gate-pass > store progress(bench_round/bench_refit/rev:sN) > fail_threshold 세션
  로테이션(기본 3; review/measure/benchmark/audit 2) > `--max-iterations`(run-until-done.sh가
  라운드당 15로 설정).
- 모델: red/impl/s0/audit = 기본 티어, review/measure/benchmark = sonnet.
- agent timeout: 기본 75m, measure 180m, benchmark 240m (tree 실행은 라운드 캐시로 중단 재개).
- lua_timeout: 기본 90s, s0/impl/measure/benchmark 900s(in-gate 전체 suite/셀프체크),
  harness-audit 3600s(regen 멱등 검사가 suite를 여러 번 돈다).
- usage limit은 gralph 밖에서 흡수: run-until-done.sh가 CLI 배너(`hit your ... limit` 등)에만
  매칭해 슬립 후 재개. bare `quota|rate limit|429` 매칭 금지(리서치 출력이 그 단어를 찍는다).
- **stall 판정은 cursor 동일성 기준**: s6-benchmark가 이전 스테이지로 되돌리는 것은 정상
  진행이므로 선형 인덱스 후퇴를 stall로 세지 않는다. 같은 cursor로 2라운드 무진전 + 배너
  없음 → STUCK 알림 후 인간에게 반환.
- **완료 알람**: run-until-done.sh가 DONE/STUCK/TIMEOUT 세 출구에서 `notify()`(터미널 벨 +
  BurntToast → `msg *` → 벨 폴백)를 호출한다. 루프는 별도 프로세스라 이것 없이는 아무것도
  완료를 알리지 않는다. Claude Code에서 `run_in_background: true`로 띄우면 종료 시 세션이
  재호출되어 그 자체가 알람이 된다. 원격 푸시가 필요하면 notify() 본문을 telegram/ntfy curl로.

## Law 6 프로브 로그 (2026-07-26, gralph v0.1.0, Windows/git-bash)

첫 실행 전 증명 완료. 프로브 아티팩트(더미 bench·프로브 테스트·evidence·.gralph)는 전부
삭제했고, `[sq][s1]`/`[sq][s2]`/`[sq][s5]` 프로브 커밋은 reset으로 제거했다(잔존 시 커밋
게이트가 영구 무력화되므로).

- 전 23노드 FAIL(빈 상태): 각자 처방형 사유로 fail-closed ✓ (Lua 크래시 0)
- s1-red: PASS(진짜 RED junit) / FAIL(passed=1 조작) ✓
- s1-impl: PASS / FAIL(테스트 1바이트 탬퍼→hash_match) / FAIL(구현 파손→stage_failed) —
  치트별 고유 메시지 ✓
- s1-review: PASS→s1-measure / PASS→s1-impl(+store rev:s1=1) / FAIL(stale head_sha) /
  FAIL(blocking_count 누락) ✓
- s1-measure: PASS→s2-red / FAIL(scraped_pages_min=2) / FAIL(커밋 없음) /
  FAIL(gptr-mcp dirty — 프로브 중 실제 오염을 검출, WIP을 `[sq][baseline]`으로 커밋해 해소) ✓
- s2-measure: FAIL(S1_min_pct=72) / **refit PASS**(store bench_refit=s2 → s6-benchmark 라우트
  + bench_refit="" 클리어) ✓
- s3-measure: FAIL(traps_hit_total=1) / s4-measure: FAIL(pruned_count_total=0) ✓
- s5-measure: PASS→s6-benchmark — **단일 successor 노드의 명시적 gralph.route() 합법 확인** ✓
- s6-benchmark: PASS→harness-audit / PASS→s1-impl(weakest=S2, store bench_round=1·
  bench_refit="s1") / FAIL(queries_scored=4 hollow zero) / FAIL(stale bench_round) ✓
- harness-audit: FAIL(try_reports_missing) ✓
- code_fp 신선도(2026-07-26 추가): FAIL(code_fp 없는 evidence → "DIFFERENT implementation
  bytes" 처방) / PASS-통과확인(올바른 code_fp를 넣으면 신선도 검사를 지나 실제 임계값
  S1_min_pct=33<80 에서 멈춤 — 검사 통과가 증명됨) ✓. code_fp 자체: 2회 연속 동일(멱등),
  구현 1바이트 수정 시 변경, 원복 시 원값 복귀 ✓
- 채점기 동결(2026-07-26 추가): FAIL(manifest의 scorer 해시 불일치 →
  `frozen_file_MODIFIED:bench/score_report.py`) / PASS(정상 manifest → frozen_ok=1) ✓
- impl ack 바인딩(2026-07-26 추가): 4방향 — stale ack(필드 없음)/틀린 review_head_sha/
  올바른 sha지만 blocking id 누락 = 전부 거부, 정상 ack만 수용 ✓
- s0 diversity(추가 하드닝, 2026-07-26): FAIL(동일 category 5개 → categories_distinct_1_lt_4,
  gralph try에서 처방 메시지 확인) / FAIL(facts 1개 → facts_lt_5:\<id\> — 골든 지목) /
  PASS(category 5종·dated 2·pair 상이·도메인 10 셋에서 diversity_ok=1) ✓
- **live에서 첫 증명되는 PASS 경로**: s0-bench(진짜 bench 없이는 정직한 PASS 불가),
  harness-audit(자기 자신의 try 매트릭스 필요), s2~s4의 red/impl/review 래퍼 PASS(증명된
  공통 모듈의 1줄 래퍼; harness-audit 스테이지가 전 노드 양방향 try 매트릭스를 재생성한다).

## 잔존 리스크 (게이트가 과대 주장하지 않도록 명시)

- review 판정은 LLM이다(사용자 지정 설계). 결정적으로 잡는 것은 신선도(head_sha)·형식·라운드
  캡뿐이고, impl_ack의 addressed_findings는 **에이전트 자기보고**다(review_head_sha 바인딩으로
  "이번 review에 대한 ack"임은 강제되지만, 내용의 진위는 아니다) — 실질 재판정은 다음 review
  라운드가 한다.
- **3라운드 캡은 인간 개입 지점이지 자동 해소되지 않는다.** 캡 도달 후에도 루프는 그 노드를
  계속 재시도하며 세션을 태운다(s2에서 실측: 캡 이후 2세션 추가 소모 후 수동 정지). 캡이
  걸리면 즉시 정지 → 지적 검증 → 아래 "개입 기록" 형식으로 판단을 남기고 `rev:sN` 리셋.
  캡 도달 상태에서 리뷰어가 압박을 받아 blocking을 minor로 강등하면 게이트가 조용히
  통과하므로, 캡 도달은 사람이 봐야 한다.
- S1 채점의 URL 재fetch는 네트워크 의존이다. fetch-cache로 재채점은 멱등이지만, 캐시 미스
  상태의 첫 fetch는 원문 변경/소실에 노출된다(캐시가 남는 한 audit의 regen_ok는 성립).
- 골든셋 facts/traps의 품질은 s0 에이전트의 검증 성실성에 달렸다. 게이트가 강제하는 것은
  스키마·픽스처 판별력·출처 존재이지, 사실의 참/거짓 자체가 아니다.
- container_probe_s1은 GPTResearcher 내부 속성(visited_urls/context)을 introspect한다 —
  업스트림 리네임 시 probe가 0을 보고하며 fail-closed로 떨어진다(조용히 통과하지 않음).

## 개입 기록 (human-in-the-loop)

캡·하드페일로 루프가 멈추면 판단 근거를 여기 남긴다. 게이트가 사람에게 넘긴 결정이므로,
"왜 라운드를 더 줬는가"가 기록되지 않으면 다음 사람은 같은 검증을 반복한다.

**2026-07-26 s2-measure 무한실패 — 캐시 키에 코드 지문 추가(하네스 결함 수정)**
- 증상: s2-measure가 8회 반복 실패. 원인은 리뷰나 구현이 아니라 **내가 만든 캐시**였다.
  `run_tree_cached`의 키가 (라운드, 골든id)뿐이라, 구현을 고치고 measure를 재실행해도
  **옛 코드가 만든 tree.json을 다시 채점**한다 → 점수가 영원히 안 움직인다 = law 6의
  "고장난 심판"(올바른 작업을 수용할 수 없는 게이트). 실측 증거: 캐시 21:22/21:38 산출,
  `tree_research.py` 21:45 수정.
- 수정: 캐시 키에 `code_fp`(gpt_researcher/**/*.py + gptr-mcp/server.py **내용** 해시)를
  포함. git sha가 아니라 내용 해시인 이유 — 에이전트는 커밋 전에 반복하므로 HEAD 기준
  키는 수정된 코드를 옛 산출물로 통과시킨다. 중단 재개(같은 코드)는 그대로 무료.
- 동시에 게이트도 강화: measure/benchmark evidence에 `code_fp`를 기록하고 게이트가
  **재계산해 대조** — 측정 후 구현을 건드리면 그 수치는 존재하지 않는 코드를 설명하므로
  거부된다.

**2026-07-26 s2-review 3라운드 캡 도달 — 라운드 리셋(rev:s2 → 0), cursor를 s2-impl로 되감음**
- 리뷰어 판정: blocking R1 (`researcher.py:822` + `tree_research.py:215-228`).
- 사람 검증 결과 **진성 결함 확정**: prefetch 분기가 `add_research_sources([{"url": url}])`로
  본문 없이 소스를 넣는다. 그런데 tree의 s2 narrowing은 `read_docs[url]`이 truthy일 때만
  `node.sources`를 유지하므로, retriever가 이미 본문을 반환한 소스(=firecrawl:81,
  pubmed_central, custom — 이 fork의 **주력 경로**)는 인용 여부와 무관하게 전량 드롭된다.
  s2가 고치려던 결함 6a(bun 실측 13노드/4인용)를 그대로 재현하는 구조. s2 단위테스트는 가짜
  `get_research_sources()`가 raw_content를 직접 넣어줘서 이 경로를 아예 타지 않는다.
- 판단: 3라운드가 **매번 서로 다른 진짜 버그**를 찾아낸 수렴 과정이지 핑퐁이 아니므로 라운드
  재부여가 타당. 코드 수정은 사람이 하지 않고 impl 노드에 맡긴다(리뷰 레인 분리 유지).
- 같은 사이클에서 발견된 게이트 구멍 2건 즉시 수정: (1) ack의 finding id 재사용 충돌 →
  `review_head_sha` 바인딩 추가, (2) `"stage":N` 토큰이 sort_keys 직렬화에서 `}` 종결일 때
  매칭 실패 → 종결자 양쪽 허용(순수 확장).

## 운영 메모

- 코드 변경 후 컨테이너 반영은 measure/benchmark 스크립트의 force-recreate가 담당. 수동
  확인은 `docker compose -f D:/docker/gptr-mcp/docker-compose.yml up -d --force-recreate
  gptr-mcp` 후 `/health` 200.
- evidence 복구는 언제나 `python scripts/regen_evidence.py --what s0|impl|measure|benchmark
  [--stage N]` — 손 편집 금지(law 10). measure/benchmark는 `no_read/bench_runs/round{R}/`
  캐시에서 재채점하므로 live 재실행 없이 멱등 재생성된다.
- cursor 되감기: `.gralph/search-quality/state.json`의 `cursor`를 직접 편집(게이트 하드닝 후
  꼬리 스테이지만 재실행할 때). `gralph try`는 dry-run이라 live 상태에 안전.
