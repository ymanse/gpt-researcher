"""RED tests for s5 — roll-up consistency against the node answers
(spec/search-quality.md, defect 6b).

Observed:
  * Nothing between the post-order roll-up and the emitted report compares the
    assembled CLAIMS against what the nodes actually found. `run()` goes body ->
    uncited-[id] strip -> ungrounded-marker strip -> Citations list; both strips
    police citation MARKERS, neither asks whether the sentence agrees with any
    node answer. A number the merge invents, or one that conflicts with a node's
    own figure, ships as a finding.
  * `_persist` writes `nodes[]` as `{id, depth, status, question}` only, so the
    S6 correspondence corpus the frozen scorer builds from `node["answer"]`
    (bench/score_report.py main(): `if node.get("answer"): corpus.append(...)`)
    is EMPTY for every tree.json ever produced — every numeric claim in the
    report then falls through to `unsupported`, the penalty saturates at 1.0 and
    S6 is 0 no matter how good the research was. The spec closes this by ADDING
    `nodes[].answer` (backward-compatible addition; the existing four keys stay).

Contract pinned here (implemented in s5-impl):
  (a) `verify_rollup(body) -> (kept_body, contradictions, unsupported)`. A claim
      whose number conflicts with a node answer covering the same ground is
      reported as a contradiction and dropped from the body — leaving it in
      means the scorer counts it, and the s5 live gate is
      `contradictions_total == 0`.
  (b) a claim no node answer stands behind is reported unsupported and dropped.
      A FAILED node's answer is prior knowledge written over missing evidence
      (defect 3 / s3): it is not evidence for anything.
  (c) claims a node answer does support, and prose carrying no figure, pass
      through byte-identical — the pass may not quietly reflow the report.
  (d) `run()` routes the assembled roll-up through that pass and reports its
      counts, and the persisted tree.json nodes[] carry `answer` (empty for a
      FAILED node, exactly as `_node_dict` already blanks its sources/digest).

The matching rule is not invented here — it is the frozen S6 scorer's, which is
what the live gate measures (bench/score_report.py score_s6, frozen since s0):
a claim sentence is SUPPORTED when some corpus text shares one of its
significant numbers AND >= min(2, |ctx|) context tokens (>=4 chars, non-numeric,
non-stopword); CONTRADICTED when a corpus text shares >= 3 context tokens and
none of its numbers; UNSUPPORTED otherwise. Every fixture below is hand-checked
against that rule.

Deterministic, no network: the unit tests need no LLM at all, and the two
end-to-end tests patch GPTResearcher / create_chat_completion at the
tree_research module seam and replace `embed_question` on the instance.
"""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod

# --------------------------------------------------------------------------
# node answers = the corpus the report is checked against.
#   NODE_A ctx: porting team moved lines rust eleven days worktree sharded
#               agent harness drove migration      nums: {530000}
#   NODE_B ctx: fuzzing security review full test suite gated every merged
#               shard adversarial reviewer worked queue compile errors
#                                                  nums: {16000}
# --------------------------------------------------------------------------
NODE_A_ANSWER = (
    "The porting team moved 530,000 lines of Zig into Rust in eleven days. "
    "A worktree-sharded agent harness drove the migration."
)
NODE_B_ANSWER = (
    "Fuzzing, security review and the full test suite gated every merged shard. "
    "The adversarial reviewer worked a queue of 16,000 compile errors."
)

TITLE = "# How did the porting team move a large Zig codebase over to Rust?"

# nums {530000} overlap NODE_A, ctx {port moved lines code rust} shares
# moved/lines/rust (3) >= min(2, 5) -> SUPPORTED
SUPPORTED = "The port moved 530,000 lines of Zig code to Rust."
# ctx identical to NODE_A's first sentence (7 shared tokens >= 3) but its number
# is 12,000 against the node's 530,000 -> CONTRADICTED
CONTRADICTING = "The porting team moved 12,000 lines of Zig into Rust in eleven days."
# shares no number and no context token with any node answer -> UNSUPPORTED
UNSUPPORTED = "Quarterly kumquat pallet shipments reached 48,500 units."
# carries no significant number at all, so it is not a claim the scorer weighs
PROSE = "The harness sharded the work across git worktrees."


def _bare_skill():
    """A skill with no LLM, no researcher, no network — verify_rollup reads
    nothing but self.nodes."""
    parent = SimpleNamespace(
        query=TITLE,
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    return tree_mod.TreeResearchSkill(parent)


def _add_node(skill, node_id, question, answer, status, depth=1, parent_id="0"):
    node = tree_mod.ResearchNode(id=node_id, question=question, parent_id=parent_id,
                                 depth=depth)
    node.status = status
    node.answer_md = answer
    node.answer_digest = answer
    node.learnings = [answer]
    skill.nodes[node_id] = node
    return node


def _researched_tree():
    """Two nodes that really researched: the corpus for (a)-(c)."""
    skill = _bare_skill()
    _add_node(skill, "0", "How did the team port Zig to Rust?", NODE_A_ANSWER,
              tree_mod.NodeStatus.EXPANDED, depth=0, parent_id=None)
    _add_node(skill, "0.0", "Which verification layers gated each shard?", NODE_B_ANSWER,
              tree_mod.NodeStatus.ANSWERED)
    return skill


async def _verify(skill, body):
    """verify_rollup may become a coroutine if it ever re-researches a node."""
    out = skill.verify_rollup(body)
    return await out if inspect.isawaitable(out) else out


# ---------------------------------------------------------------------------
# (a) a claim that contradicts a node answer
# ---------------------------------------------------------------------------

async def test_claim_contradicting_a_node_answer_is_detected_and_dropped():
    skill = _researched_tree()
    body = "\n".join([TITLE, "", SUPPORTED + " [1]", CONTRADICTING, ""])

    clean, contradictions, unsupported = await _verify(skill, body)

    assert any("12,000" in c for c in contradictions), (
        "a node answer says the port moved 530,000 lines; this claim says 12,000 "
        "over the same ground. Nothing in run() compares the merged claims "
        f"against the node answers today, so it ships. got contradictions="
        f"{contradictions!r}"
    )
    assert not any("12,000" in u for u in unsupported), (
        "the tree DOES cover this ground — it is a conflict, not a hole. The "
        "scorer separates the two counts and the s5 gate reads both"
    )
    assert CONTRADICTING not in clean, (
        "detection alone leaves the claim in the report, where score_report.py "
        "still counts it: the s5 live gate is contradictions_total == 0"
    )
    assert SUPPORTED + " [1]" in clean, (
        "the consistent claim next to it, and its citation marker, must survive"
    )
    assert TITLE in clean, "the report heading is not a claim"


# ---------------------------------------------------------------------------
# (b) a claim no node answer supports
# ---------------------------------------------------------------------------

async def test_claim_no_node_answer_supports_is_reported_unsupported_and_dropped():
    """The FAILED node here states the claim verbatim: a starved node answers
    from prior knowledge (defect 3), so treating its text as evidence would
    launder exactly the fabrication s3 fails closed on back into the report."""
    skill = _researched_tree()
    _add_node(skill, "0.1", "Which vendors ship a marmalade cartography module?",
              UNSUPPORTED, tree_mod.NodeStatus.FAILED)
    body = "\n".join([TITLE, "", SUPPORTED, UNSUPPORTED, ""])

    clean, contradictions, unsupported = await _verify(skill, body)

    assert any("48,500" in u for u in unsupported), (
        "no researched node answer shares a number or context with this claim. "
        "The only node that 'covers' it is FAILED — prior-knowledge text over "
        f"missing evidence, never evidence. got unsupported={unsupported!r}"
    )
    assert not any("48,500" in c for c in contradictions), (
        "nothing in the tree disagrees with it — the tree simply never found it"
    )
    assert UNSUPPORTED not in clean, (
        "spec s5: an unsupported claim is removed (or its node re-researched); "
        "the live gate is unsupported_claims_total == 0"
    )
    assert SUPPORTED in clean and TITLE in clean


# ---------------------------------------------------------------------------
# (c) no regression: supported claims and plain prose are untouched
# ---------------------------------------------------------------------------

async def test_supported_claims_and_prose_pass_through_unchanged():
    skill = _researched_tree()
    body = "\n".join([TITLE, "", SUPPORTED + " [1]", PROSE, "",
                      "## Verification", "", NODE_B_ANSWER, ""])

    clean, contradictions, unsupported = await _verify(skill, body)

    assert contradictions == [] and unsupported == [], (
        "every figure here traces to a node answer (530,000 to node 0, 16,000 "
        "to node 0.0) and the prose carries no figure at all. Flagging these "
        "would gut real findings out of the report — the pass has to be a "
        f"filter, not a shredder. got {contradictions!r} / {unsupported!r}"
    )
    assert clean == body, (
        "a report that survives the check must come back byte-identical: "
        "markdown structure (headings, blank lines) and [id] markers are what "
        "the S1 grounding scorer reads, so a pass that re-joins sentences it "
        "kept silently rewrites the artifact it was only asked to check"
    )


# ---------------------------------------------------------------------------
# (d) run() wires the pass in, and tree.json carries the node answers
# ---------------------------------------------------------------------------

ROOT_Q = "How did the porting team move a large Zig codebase over to Rust?"
CHILD_Q = "Which verification layers gated each ported shard?"
STARVED_Q = "Which vendors ship a marmalade cartography stalactite module?"

ROOT_A = "The team moved 530,000 lines of Zig into Rust in eleven days."
CHILD_A = "Fuzzing, security review and the full test suite gated every merged shard."
# a starved node's answer: written from prior knowledge, never from evidence
STARVED_A = "Zorbulate framistan sprockets emitted 4.2 gigawatts during the port."
ANSWER_BY_Q = {ROOT_Q: ROOT_A, CHILD_Q: CHILD_A, STARVED_Q: STARVED_A}

CONTEXT_MARK = "Collected page text from the scraped sources."
RICH_CONTEXT = (CONTEXT_MARK + " ") * 400          # ~18k chars, over MIN_CONTEXT_CHARS
THIN_CONTEXT = CONTEXT_MARK + " Only one fragment survived."  # under it -> FAILED

EMBEDDINGS = {
    ROOT_Q: [1.0, 0.0, 0.0, 0.0],
    CHILD_Q: [0.0, 1.0, 0.0, 0.0],
    STARVED_Q: [0.0, 0.0, 1.0, 0.0],
}
DEFAULT_EMBEDDING = [0.0, 0.0, 0.0, 1.0]


def _llm_response(answer: str) -> str:
    return f"ANSWER: {answer}\nDIGEST: {answer}\nLEARNINGS:\n- {answer}\n"


def _make_skill(monkeypatch):
    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.query = query
            self.visited_urls = visited_urls if visited_urls is not None else set()
            self._answer = ANSWER_BY_Q.get(query, "Unmapped question answer.")
            self._url = f"https://example.com/{abs(hash(query)) % 10000}"

        async def conduct_research(self):
            self.visited_urls.add(self._url)
            # the starved node still READS a document — it is the thin context,
            # not an empty hand, that must fail it closed (research_node's
            # MIN_CONTEXT_CHARS branch), so its answer text really exists
            return THIN_CONTEXT if self.query == STARVED_Q else RICH_CONTEXT

        def get_research_sources(self):
            return [{"url": self._url, "title": "doc",
                     "raw_content": (self._answer + " Supporting prose. ") * 8}]

        def get_costs(self):
            return 0.0

    async def fake_chat(messages=None, **kwargs):
        content = " ".join(str(m.get("content", "")) for m in (messages or []))
        if CONTEXT_MARK in content:  # the ANSWER prompt
            for question, answer in ANSWER_BY_Q.items():
                if question in content:
                    return _llm_response(answer)
            return _llm_response("Unmapped question answer.")
        if ROOT_Q in content:        # the EXPANSION prompt
            return f"Question: {CHILD_Q}\nQuestion: {STARVED_Q}"
        return ""

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    parent = SimpleNamespace(
        query=ROOT_Q,
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    skill = tree_mod.TreeResearchSkill(parent)

    async def fake_embed(text):
        return list(EMBEDDINGS.get(text, DEFAULT_EMBEDDING))

    skill.embed_question = fake_embed
    return skill


async def test_run_routes_the_assembled_rollup_through_the_consistency_pass(monkeypatch):
    """Seam test (the module's own convention: run() must route through the
    instance). A pass nobody calls detects nothing — that is the whole defect."""
    skill = _make_skill(monkeypatch)
    seen = []

    def spy(body):
        seen.append(body)
        return body + "\n\nCONSISTENCY-PASS-OUTPUT\n", ["one contradiction"], ["u1", "u2"]

    skill.verify_rollup = spy
    result = await skill.run(query=ROOT_Q, max_depth=1, max_breadth=2, max_nodes=10)

    assert len(seen) == 1, (
        "run() must hand the assembled roll-up to verify_rollup exactly once "
        f"(got {len(seen)} calls)"
    )
    assert "530,000 lines" in seen[0], (
        "the pass must see the rolled-up report body, not a fragment: the claims "
        "it checks only exist after post-order synthesis merged the nodes"
    )
    assert "CONSISTENCY-PASS-OUTPUT" in result["report_md"], (
        "the body the pass returns is what gets reported — otherwise the check "
        "runs and its result is thrown away"
    )
    assert result["stats"]["contradictions"] == 1, (
        "the counts must reach the caller; the s5 live measure reads them back "
        "out of the scored run and a silent pass leaves nothing to read"
    )
    assert result["stats"]["unsupported_claims"] == 2


async def test_tree_json_nodes_carry_answer_text(monkeypatch, tmp_path):
    skill = _make_skill(monkeypatch)

    result = await skill.run(query=ROOT_Q, max_depth=1, max_breadth=2, max_nodes=10,
                             outputs_dir=str(tmp_path))
    raw = Path(result["artifacts"]["tree_json"]).read_text(encoding="utf-8")
    payload = json.loads(raw)
    nodes = payload["nodes"]

    assert isinstance(nodes, list), (
        "the frozen scorer iterates `for node in tree.get('nodes', [])` and calls "
        "node.get('answer') — a dict here makes it iterate id STRINGS and crash"
    )
    for node in nodes:
        for key in ("id", "depth", "status", "question"):
            assert key in node, (
                f"tree.json's existing contract keeps {key!r} — s5 ADDS a field, "
                "it does not reshape the artifact (spec: 하위호환 추가만)"
            )
        assert "answer" in node, (
            "spec: 's5 이후 노드 답변 텍스트가 채점에 필요하므로 nodes[].answer 를 "
            "추가한다'. Without it the S6 corpus is empty and every numeric claim "
            f"in the report scores unsupported. missing on node {node.get('id')!r}"
        )

    by_question = {n["question"]: n for n in nodes}
    assert "530,000 lines" in by_question[ROOT_Q]["answer"], (
        "a researched node's answer text is what the report's claims are matched "
        "against — the digest-only or empty string grounds nothing"
    )
    assert "security review" in by_question[CHILD_Q]["answer"]

    starved = by_question[STARVED_Q]
    assert starved["status"] == "failed", (
        "sanity: the thin-context node must still fail closed (s3)"
    )
    assert starved["answer"] == "", (
        "a FAILED node's answer is prior knowledge over missing evidence. "
        "_node_dict already blanks its sources and digest for exactly this "
        "reason; emitting it here would let the starved text back in as the "
        "corpus that validates report claims — and a `or n.question` fallback "
        "would smuggle the unresearched question in just as badly"
    )
    assert "zorbulate" not in raw.lower(), (
        "no trace of the starved node's fabricated answer may reach tree.json"
    )
