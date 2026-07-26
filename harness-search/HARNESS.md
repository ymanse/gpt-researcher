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
| **sN-impl** | GREEN 안티탬퍼: RED 파일 sha256 재계산 일치(hash_match), collected 비감소, 스테이지+누적 전체 suite green(failed/errors/skipped=0), ruff_errors=0(E9,F — 변경 파일 한정), frozen_ok, review_addressed_ok(직전 review의 blocking id 전부가 impl_ack의 addressed_findings에 존재) | **in-gate `verify_impl.py` 재실행** — 위조 evidence는 생존 불가 |
| **sN-review** | 분리 레인 적대 리뷰: 입력은 `review_diff.py`의 diff + 스펙 완료조건뿐. head_sha가 현재 HEAD와 일치(stale 리뷰 재활용 차단, in-gate 재계산), findings 배열 + blocking_count 숫자. blocking_count>0 → `sN-impl`로 라우트(store `rev:sN` 카운트, **3라운드 초과 시 hard fail**), 0 → `sN-measure` | 판정 자체는 LLM(의도된 law 2 예외, 아래 잔존 리스크) — 신선도·형식·라운드 캡은 결정적 |
| **sN-measure** | 실컨테이너 측정: recreated:true + health:200(bind-mount 반영), bench_round가 store와 일치(stale evidence 차단), 스테이지 임계값(아래), frozen_ok, `[sq][sN]` 커밋 + 양 repo 클린. refit 라운드면 s6-benchmark로 직행 라우트(bench_refit 클리어) | `measure.py`만이 evidence를 쓴다; 라운드 캐시로 멱등 |
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
- s0 diversity(추가 하드닝, 2026-07-26): FAIL(동일 category 5개 → categories_distinct_1_lt_4,
  gralph try에서 처방 메시지 확인) / FAIL(facts 1개 → facts_lt_5:\<id\> — 골든 지목) /
  PASS(category 5종·dated 2·pair 상이·도메인 10 셋에서 diversity_ok=1) ✓
- **live에서 첫 증명되는 PASS 경로**: s0-bench(진짜 bench 없이는 정직한 PASS 불가),
  harness-audit(자기 자신의 try 매트릭스 필요), s2~s4의 red/impl/review 래퍼 PASS(증명된
  공통 모듈의 1줄 래퍼; harness-audit 스테이지가 전 노드 양방향 try 매트릭스를 재생성한다).

## 잔존 리스크 (게이트가 과대 주장하지 않도록 명시)

- review 판정은 LLM이다(사용자 지정 설계). 결정적으로 잡는 것은 신선도(head_sha)·형식·라운드
  캡뿐이고, impl_ack의 addressed_findings는 **에이전트 자기보고**다 — 실질 재판정은 다음
  review 라운드가 한다.
- S1 채점의 URL 재fetch는 네트워크 의존이다. fetch-cache로 재채점은 멱등이지만, 캐시 미스
  상태의 첫 fetch는 원문 변경/소실에 노출된다(캐시가 남는 한 audit의 regen_ok는 성립).
- 골든셋 facts/traps의 품질은 s0 에이전트의 검증 성실성에 달렸다. 게이트가 강제하는 것은
  스키마·픽스처 판별력·출처 존재이지, 사실의 참/거짓 자체가 아니다.
- container_probe_s1은 GPTResearcher 내부 속성(visited_urls/context)을 introspect한다 —
  업스트림 리네임 시 probe가 0을 보고하며 fail-closed로 떨어진다(조용히 통과하지 않음).

## 운영 메모

- 코드 변경 후 컨테이너 반영은 measure/benchmark 스크립트의 force-recreate가 담당. 수동
  확인은 `docker compose -f D:/docker/gptr-mcp/docker-compose.yml up -d --force-recreate
  gptr-mcp` 후 `/health` 200.
- evidence 복구는 언제나 `python scripts/regen_evidence.py --what s0|impl|measure|benchmark
  [--stage N]` — 손 편집 금지(law 10). measure/benchmark는 `no_read/bench_runs/round{R}/`
  캐시에서 재채점하므로 live 재실행 없이 멱등 재생성된다.
- cursor 되감기: `.gralph/search-quality/state.json`의 `cursor`를 직접 편집(게이트 하드닝 후
  꼬리 스테이지만 재실행할 때). `gralph try`는 dry-run이라 live 상태에 안전.
