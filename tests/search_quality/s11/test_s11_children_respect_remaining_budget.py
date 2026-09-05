"""s11 (P1.4): children are generated only for nodes the budget can actually research.

Measured over the last 5 production tree runs (harness-search/spec/p0-p1-slimming.md):
67-80% of every node the tree CREATED was still PENDING when the run ended. Each of
those nodes cost a `generate_child_questions` LLM call plus a question embedding, and
none of them was ever researched — the expansion kept inventing follow-ups long after
the node budget, the CLI-session allowance and the clock could no longer pay for one
more node.

Contract pinned here (spec corrected 2026-09-05 — the rule is ONLY the gate):

  (a) capacity is the MINIMUM over the limits that apply, expressed in NODES (nodes
      left, agent calls left minus the synthesis reserve, wall clock left).
  (b) `capacity - len(frontier) <= 0` skips the `generate_child_questions` call
      ENTIRELY. Not making the call is the whole saving.
  (c) a node that IS expanded accepts up to `max_breadth` children and nothing else
      clamps it — byte-identical to today. Accepted-but-unresearched nodes stay
      PENDING on purpose: they are what `assemble_report` publishes under
      "## Unresearched Questions". Clamping the accepted count to the remaining
      capacity was tried and measured — it took the suite 158 -> 156, deleting s4's
      primary-source node and the whole disclosure section.
  (d) the nodes term and the calls term each bind on their own; an unbounded run
      (`agent_budget_limit() == 0`) is never throttled by the calls term, and a run
      whose per-node call cost measures 0 is not throttled either — an unmeasurable
      cost is "not limiting", never "capacity 0".
  (e) nodes that ARE expanded still create and PRUNE exactly as today (s4/s8).

Deterministic: research / expansion / embedding are replaced on the instance (the seams
`TreeResearchSkill` documents as test seams), `create_chat_completion` is patched at the
module, and the four budget helpers are patched where `tree_research` imported them —
including `agent_budget_exhausted`, so a stale process-wide allowance armed by another
test cannot end the batch loop before the expansion decision is even reached. No
network, no LLM, no embeddings service.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import gpt_researcher.skills.tree_research as tr


ROOT_Q = "What are the operational failure modes of vector-index rebuilds at scale?"
# cos(NEAR_Q, ROOT_Q) == 0.8 -> novelty 0.20, under the 0.30 floor, but still under
# DEDUP_COSINE (0.92), so it is really CREATED and then PRUNED (the s4/s8 contract).
NEAR_Q = "Which operational failures show up when a vector index is rebuilt at scale?"

DIM = 64
# axes 0 and 1 are reserved for the fixed table below; every other question gets its
# own axis on demand, so unrelated questions are orthogonal -> novelty 1.0, no dedup.
FIXED_EMB = {
    ROOT_Q: [1.0, 0.0],
    NEAR_Q: [0.8, 0.6],
}

# CLI sessions one researched node spawns in these tests. The spec's own measurement
# is 8-9 per node and its gate target is <= 3; 3 is the friendlier of the two for an
# implementation that derives calls/node from the measured spend delta.
CALLS_PER_NODE = 3


def _pad(vec):
    return list(vec) + [0.0] * (DIM - len(vec))


class _Tree:
    """One hermetic tree run: fakes every LLM/embedding seam and COUNTS the calls.

    Counting is what this stage is about — a children call whose output is then
    discarded costs exactly as much as one whose children are kept.

    `calls_charged=False` models a run whose fakes spawn no CLI session at all, so the
    measured per-node call cost is 0. That is the harness's own normal state, and it
    must read as "the calls term cannot be evaluated", not as "no node is affordable".
    """

    def __init__(self, monkeypatch, *, limit, spent, reserve,
                 child_questions=None, offered=4, calls_charged=True):
        self.researched = []       # node ids handed to research_node
        self.expanded = []         # node ids handed to generate_child_questions
        self.calls_spent = spent   # CLI sessions charged so far in this process
        self._offered = offered
        self._child_questions = child_questions
        self._calls_charged = calls_charged
        self._axes = {}
        self._next_axis = len(FIXED_EMB)

        parent = SimpleNamespace(
            query=ROOT_Q,
            cfg=SimpleNamespace(strategic_llm_provider="mock",
                                strategic_llm_model="mock", config_path=None),
            tone=None,
            websocket=None,
            headers={},
            visited_urls=set(),
        )
        async def _chat(*a, **kw):
            return "ANSWER: x\nDIGEST: x\nLEARNINGS:\n- x\n"

        # patched BEFORE the skill is constructed: an implementation that snapshots the
        # spend at construction to measure calls/node later must see the fake counter
        monkeypatch.setattr(tr, "create_chat_completion", _chat)
        monkeypatch.setattr(tr, "agent_budget_limit", lambda: limit)
        monkeypatch.setattr(tr, "agent_calls_spent", lambda: self.calls_spent)
        monkeypatch.setattr(tr, "agent_synthesis_reserve", lambda: reserve)
        # the batch loop's own stop condition is not what is under test here, and it
        # reads process-wide globals another test may have armed
        monkeypatch.setattr(tr, "agent_budget_exhausted", lambda reserve=0: False)

        self.skill = tr.TreeResearchSkill(parent)
        self.skill.research_node = self._research
        self.skill.generate_child_questions = self._children
        self.skill.embed_question = self._embed

    async def _research(self, node):
        self.researched.append(node.id)
        if self._calls_charged:
            self.calls_spent += CALLS_PER_NODE
        node.status = tr.NodeStatus.ANSWERED
        node.answer_md = node.answer_digest = f"findings for {node.question}"
        node.learnings = [f"finding from {node.id}"]

    async def _children(self, node):
        self.expanded.append(node.id)
        if self._calls_charged:
            self.calls_spent += 1  # the children call is itself one CLI session
        if self._child_questions is not None:
            return list(self._child_questions)
        return [f"{node.id} follow-up {i}" for i in range(self._offered)]

    async def _embed(self, text):
        if text in FIXED_EMB:
            return _pad(FIXED_EMB[text])
        if text not in self._axes:
            self._axes[text] = self._next_axis
            self._next_axis += 1
        vec = [0.0] * DIM
        vec[self._axes[text] % DIM] = 1.0
        return vec

    async def run(self, **kwargs):
        # the clock is never the binding limit here: the seconds/node estimate is left
        # unpinned by these tests, so it is opened wide and the other terms decide.
        params = dict(query=ROOT_Q, max_depth=3, max_breadth=4, max_nodes=3,
                      node_concurrency=1, time_budget_s=1e6)
        params.update(kwargs)
        return await self.skill.run(**params)


def _children_of(result, node_id="0"):
    return result["tree"]["nodes"][node_id]["children"]


@pytest.mark.asyncio
async def test_the_nodes_term_alone_stops_the_children_call(monkeypatch):
    """The nodes term binds on its own, and the saving is the SKIPPED CALL.

    Unbounded allowance and a 1e6-second clock, so `max_nodes` is the only limit.
    max_nodes=3, one node per batch: the root is researched (1 of 3), two nodes of
    capacity remain, nothing is queued, so the root IS expanded and accepts its full
    max_breadth=4 — unchanged from today, and the two it can never reach are the
    PENDING questions the report discloses. When the first child is then researched
    (2 of 3), one node of capacity is left and the frontier already holds three, so
    nothing that expansion could invent will ever be researched.
    `generate_child_questions` must not be called for it at all.
    """
    t = _Tree(monkeypatch, limit=0, spent=0, reserve=0)  # unbounded: only nodes bind
    result = await t.run()

    assert len(t.researched) == 3, (
        f"non-vacuity: the run must still spend its whole node budget, "
        f"researched={t.researched}")
    assert t.expanded == ["0"], (
        f"generate_child_questions was called for {t.expanded[1:]} as well as the root. "
        f"At those calls the frontier already held every node the remaining budget can "
        f"reach, so each one bought an LLM call plus an embedding per candidate for "
        f"children that can only end the run PENDING — the 67-80% pending share "
        f"measured across the last 5 runs")
    assert len(_children_of(result)) == 4, (
        f"the root accepted {len(_children_of(result))} children instead of "
        f"max_breadth=4: the gate clamped the accepted COUNT to the remaining capacity "
        f"(2). That saves three embeddings, not a session, and deletes the "
        f"'## Unresearched Questions' the run is supposed to disclose")


@pytest.mark.asyncio
async def test_a_call_allowance_spent_down_to_the_reserve_stops_the_children_call(
        monkeypatch):
    """The calls term binds on its own, with node budget and clock to spare.

    limit 35, 26 sessions already spent, +3 for researching the root = 29, synthesis
    reserve 5: exactly ONE call is left above the reserve the roll-up's judge lives on.
    A further node costs 3 (measured this run), so the calls term buys 0 nodes and the
    children call must not be made — even though 19 nodes of budget and the whole
    clock are still unspent.
    """
    t = _Tree(monkeypatch, limit=35, spent=26, reserve=5)
    result = await t.run(max_nodes=20)

    assert t.researched[:1] == ["0"], (
        f"non-vacuity: the root must still be researched, researched={t.researched[:5]}")
    assert t.expanded == [], (
        f"generate_child_questions was called {len(t.expanded)} time(s), first for "
        f"{t.expanded[:3]}, with 1 agent call left above the synthesis reserve and a "
        f"measured cost of {CALLS_PER_NODE} calls per node: no child it returns can "
        f"ever be researched, and the call itself eats into the reserve the merge "
        f"judge needs")
    assert result["stats"]["nodes_total"] == 1, (
        f"the tree holds {result['stats']['nodes_total']} nodes; with the expansion "
        f"call skipped the root is the only node that can exist")


@pytest.mark.asyncio
async def test_a_per_node_call_cost_that_measures_zero_is_not_limiting(monkeypatch):
    """0 calls/node means "cannot be measured", NOT "no node is affordable".

    Same allowance state as the test above at the moment of the decision — limit 35,
    29 spent, reserve 5, so exactly 1 call is left above the reserve — but these fakes
    charge NOTHING, so the measured per-node cost is 0. The run is sized so that the
    two readings of a 0 cost disagree at the SECOND decision, not just the first:

      root researched   -> 3 nodes left, frontier empty
                           not-limiting: 3 - 0 > 0        -> expand (2 kids)
                           `1 // max(0, 1)` = 1 node:  1 - 0 > 0 -> expand too
      0.0 researched    -> 2 nodes left, frontier holds 0.1
                           not-limiting: 2 - 1 = 1 > 0    -> EXPAND
                           `1 // max(0, 1)` = 1 node:  min(2, 1) - 1 = 0 -> SKIP
      0.1 researched    -> 1 node left, frontier holds 0.0's 2 kids -> skip
      0.0.0 researched  -> node budget spent, run ends

    (Scheduling verified against the un-gated code: nodes are researched in the order
    0, 0.0, 0.1, 0.0.0.) So `expanded == ["0", "0.0"]` is reachable only by treating an
    unmeasurable cost as "not limiting". A `max(cost, 1)` floor buys 1 node, skips 0.0
    and expands 0.1 instead (`["0", "0.1"]`); a seed constant of 2+ buys 0 nodes and
    skips even the root (`[]`).

    Without this test the cheapest way to pass every other test in this file is to skip
    every expansion in every run, which also passes as "no wasted call".
    """
    t = _Tree(monkeypatch, limit=35, spent=29, reserve=5, calls_charged=False)
    result = await t.run(max_nodes=4, max_breadth=2)

    assert t.expanded[:1] == ["0"], (
        f"the root was never expanded (expanded={t.expanded}) on a run where no fake "
        f"charges a call: 0 measured calls per node was read as capacity 0 rather than "
        f"as an unmeasurable cost, so a bounded run skips every expansion it has the "
        f"nodes and the clock to pay for")
    assert t.expanded == ["0", "0.0"], (
        f"expanded={t.expanded}: with a per-node call cost that measures 0 the calls "
        f"term cannot be evaluated and must not bind, so the nodes term alone decides "
        f"— it clears for the root and for 0.0 (2 nodes left, 1 queued) and for nobody "
        f"after them")
    assert len(_children_of(result)) == 2, (
        f"the root accepted {len(_children_of(result))} children instead of "
        f"max_breadth=2 — the accepted count must not be clamped by capacity")


@pytest.mark.asyncio
async def test_an_unbounded_run_is_not_throttled_by_the_calls_term(monkeypatch):
    """`agent_budget_limit() == 0` means unbounded, not "zero calls left".

    500 sessions spent against limit 0 is an ordinary long-lived process. Read as a
    number, `limit - spent - reserve` is -500 and would shut expansion down entirely;
    the nodes term (room for 2 more, nothing queued) is the only one that applies.
    """
    t = _Tree(monkeypatch, limit=0, spent=500, reserve=0)
    result = await t.run(max_nodes=3, max_breadth=4)

    kids = _children_of(result)
    assert kids, (
        "an unbounded run accepted no children at all — the calls term was applied to a "
        "run that has no call limit, so `0 - agent_calls_spent()` throttled it to "
        "nothing")
    assert t.expanded == ["0"], (
        f"expanded={t.expanded}: with no call limit the nodes term is the only gate, "
        f"and it clears for the root (2 nodes of capacity, empty frontier) and for "
        f"nobody after it")
    assert len(kids) == 4, (
        f"the root accepted {len(kids)} children; a node that passes the gate accepts "
        f"up to max_breadth=4 and is clamped by nothing else")


@pytest.mark.asyncio
async def test_a_node_with_capacity_still_accepts_only_max_breadth_children(monkeypatch):
    """Regression guard (green today): max_breadth stays the ONLY cap on the count.

    max_nodes=50 leaves 49 nodes of capacity after the root, so a capacity-derived cap
    would accept all 5 candidates. max_breadth=2 must still win, and the 3 candidates
    beyond it must still be dropped without a node.
    """
    t = _Tree(monkeypatch, limit=0, spent=0, reserve=0, offered=5)
    result = await t.run(max_nodes=50, max_depth=1, max_breadth=2)

    assert t.expanded == ["0"], (
        f"only the root sits below max_depth=1, so exactly one expansion is due, "
        f"expanded={t.expanded}")
    assert len(_children_of(result)) == 2, (
        f"the root accepted {len(_children_of(result))} of 5 candidates with 49 nodes "
        f"of capacity available; max_breadth=2 must remain the only cap on the "
        f"accepted count")


@pytest.mark.asyncio
async def test_an_expanded_node_still_creates_and_prunes_its_children_as_today(
        monkeypatch):
    """Regression guard (green today): P1.4 must not touch the nodes it DOES expand.

    Budget wide open, so the gate never binds. The near-duplicate child is still
    CREATED (cosine 0.80 is under DEDUP_COSINE) and still lands as PRUNED without
    being researched — s4's create-then-prune contract and s8's prune-before-research
    contract, unchanged.
    """
    far = [f"unrelated follow-up {i}" for i in range(3)]
    t = _Tree(monkeypatch, limit=0, spent=0, reserve=0,
              child_questions=[NEAR_Q, *far])
    result = await t.run(max_nodes=50, max_depth=1, max_breadth=4)

    assert t.expanded == ["0"], (
        f"only the root sits below max_depth=1, so exactly one expansion is due, "
        f"expanded={t.expanded}")
    assert len(_children_of(result)) == 4, (
        f"all four candidates must still become nodes when the budget can pay for them, "
        f"got {len(_children_of(result))}")
    assert result["tree"]["meta"]["pruned_count"] == 1, (
        f"the near-duplicate child must still be pruned, pruned_count="
        f"{result['tree']['meta']['pruned_count']}")
    statuses = {n["question"]: n["status"] for n in result["tree"]["nodes"].values()}
    assert statuses[NEAR_Q] == "pruned", (
        f"the near-duplicate must be created and land as PRUNED, got {statuses[NEAR_Q]}")
    researched_qs = {result["tree"]["nodes"][nid]["question"] for nid in t.researched}
    assert NEAR_Q not in researched_qs, (
        "the near-duplicate was researched before being pruned — s8's "
        "prune-before-research contract")
    assert set(far) <= researched_qs, (
        f"every novel sibling must still be researched, researched={researched_qs}")
