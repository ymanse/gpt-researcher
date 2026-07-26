"""Stage 6 RED tests — deep_tree_research: persisted tree + best-first frontier.

Target module: gpt_researcher/skills/tree_research.py (does not exist yet).
Imports happen INSIDE tests so the missing implementation is a test FAILURE,
not a collection error (RED gate requires errors=0, failed>=1).

Contract pinned here (GREEN must satisfy exactly this):

- NodeStatus: Enum with PENDING/RESEARCHING/ANSWERED/EXPANDED/PRUNED/FAILED,
  .value is the lowercase string; the persisted tree serializes status as that
  string.
- ResearchNode(id, question, parent_id, depth): dataclass-style; defaults
  status=PENDING, children=[], sources=[], learnings=[], priority settable.
  sources is a list of URL strings.
- Frontier: push(node) / pop() / __len__; pop returns the highest node.priority
  first regardless of insertion order (best-first).
- TreeResearchSkill(researcher) with seam methods run() consults, so tests
  stay deterministic by replacing them on the instance:
    async research_node(node)              -> mutates node (GPTResearcher call)
    async generate_child_questions(node)   -> list[str] (Self-Ask expansion)
    async embed_question(text)             -> list[float]
    compute_novelty(node)                  -> float          (sync)
    async synthesize_node(node, ...)       -> str  (called once per non-pruned
                                                    node, post-order, root last)
- research_node itself uses GPTResearcher (imported into the module namespace,
  patchable as gpt_researcher.skills.tree_research.GPTResearcher) with
  query=node.question, awaits conduct_research(), and folds visited_urls into
  node.sources; digests may use create_chat_completion (also module-level).
- async run(query=None, max_depth=3, max_breadth=4, max_nodes=40,
            token_budget=..., credit_budget=..., novelty_threshold=0.30,
            expansion_policy="best_first", stream=False) -> dict with keys
  report_md / tree / citation_map / stats.
  tree = {"nodes": {id: {"question","status","children","parent_id","depth",
  "sources", ...}}, ...}; accepted child questions become nodes immediately, so
  when the budget (max_nodes = researched-node cap, checked before frontier
  pop) runs out, the un-researched nodes remain in the tree as "pending" and a
  synthesis is still produced.
- Dedup: a candidate question whose embedding has cosine >= 0.92 against any
  existing question embedding in the tree is dropped (no node created);
  < 0.92 is kept. Prune: after novelty is known, novelty < novelty_threshold
  marks the node PRUNED and it is never expanded; novelty == threshold
  survives.
- citation_map: {stable_id_str: url} covering exactly the URL union of all
  node sources; identical across identical runs (stable ids).
"""
import asyncio
import inspect
from types import SimpleNamespace
from unittest import mock

from gpt_researcher.config import Config

ROOT_Q = "solid-state battery supply chain bottlenecks"
C0_Q = "Which manufacturing steps limit solid-state battery throughput?"
C1_Q = "Who are the key sulfide electrolyte suppliers?"
C00_Q = "What yields do pilot lines report for sulfide cells?"
LOW_Q = "What is a battery?"
DUP_A_Q = "How do tariffs affect battery supply chains?"
DUP_B_Q = "What is the impact of tariffs on battery supply chains?"
NEAR_Q = "Do tariffs change battery sourcing decisions?"
DIST_Q = "Which startups build anode-free cells?"

URL_A = "https://a.example.com/bottlenecks"
URL_B = "https://b.example.org/suppliers"
URL_C = "https://c.example.net/yields"


def _tr():
    from gpt_researcher.skills import tree_research
    return tree_research


def _basis(i, dim=8):
    v = [0.0] * dim
    v[i] = 1.0
    return v


# cos(DUP_A, DUP_B) = 0.95 (>= 0.92 -> dropped), cos(DUP_A, NEAR) = 0.90 (kept)
_EMB = {
    ROOT_Q: _basis(0),
    C0_Q: _basis(1),
    C1_Q: _basis(2),
    C00_Q: _basis(3),
    LOW_Q: _basis(4),
    DIST_Q: _basis(5),
    DUP_A_Q: [1.0, 0.0] + [0.0] * 6,
    DUP_B_Q: [0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31224989991991992, 0.0],
    NEAR_Q: [0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4358898943540674],
}


class _FakeResearcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.query = ROOT_Q
        self.websocket = None
        self.tone = None
        self.headers = {}
        self.visited_urls = set()
        self.log_handler = None

    def get_costs(self):
        return 0.0


def _run_tree(children, novelty, sources, seconds_per_node=0.0, **run_kwargs):
    """Run the skill with every seam replaced; return (result, skill, rec).

    seconds_per_node drives a fake clock (module-local `time` is swapped, so the
    real one is untouched) — 0.0 freezes it, which is what every non-time test wants.
    """
    tr = _tr()
    skill = tr.TreeResearchSkill(_FakeResearcher(Config()))
    rec = {"expanded": [], "synth": []}
    clock = {"t": 1000.0}

    async def research(node):
        clock["t"] += seconds_per_node
        node.answer_md = f"## Answer\n\n{node.question}"
        node.answer_digest = f"digest: {node.question}"
        node.learnings = [f"learning: {node.question}"]
        node.sources = list(sources.get(node.question, []))
        node.status = tr.NodeStatus.ANSWERED

    async def children_of(node):
        rec["expanded"].append(node.question)
        return list(children.get(node.question, []))

    async def embed(text):
        return list(_EMB[text])

    def novelty_of(node):
        return novelty.get(node.question, 1.0)

    async def synth(node, *args, **kwargs):
        rec["synth"].append(node.id)
        return f"[synthesis:{node.question}]"

    skill.research_node = research
    skill.generate_child_questions = children_of
    skill.embed_question = embed
    skill.compute_novelty = novelty_of
    skill.synthesize_node = synth

    with mock.patch.object(tr, "time", SimpleNamespace(time=lambda: clock["t"])):
        result = asyncio.run(skill.run(query=ROOT_Q, **run_kwargs))
    return result, skill, rec


def _nodes(result):
    return result["tree"]["nodes"]


def _by_question(result, q):
    matches = [n for n in _nodes(result).values() if n["question"] == q]
    assert len(matches) == 1, f"expected exactly one node for {q!r}, got {len(matches)}"
    return matches[0]


def _questions(result):
    return {n["question"] for n in _nodes(result).values()}


# ---------------------------------------------------------------------------
# module contract
# ---------------------------------------------------------------------------

class TestContract:
    def test_module_exports(self):
        tr = _tr()
        for name in ("PENDING", "RESEARCHING", "ANSWERED", "EXPANDED", "PRUNED", "FAILED"):
            assert hasattr(tr.NodeStatus, name), f"NodeStatus.{name} missing"
        assert tr.NodeStatus.PRUNED.value == "pruned"
        assert tr.NodeStatus.PENDING.value == "pending"
        for m in ("run", "research_node", "generate_child_questions",
                  "embed_question", "compute_novelty", "synthesize_node"):
            assert callable(getattr(tr.TreeResearchSkill, m, None)), (
                f"TreeResearchSkill.{m} must exist (deterministic test seam)"
            )

    def test_research_node_defaults(self):
        tr = _tr()
        node = tr.ResearchNode(id="0", question=ROOT_Q, parent_id=None, depth=0)
        assert node.status == tr.NodeStatus.PENDING
        assert node.children == []
        assert node.sources == []

    def test_run_signature(self):
        tr = _tr()
        params = inspect.signature(tr.TreeResearchSkill.run).parameters
        for name in ("query", "max_depth", "max_breadth", "max_nodes",
                     "token_budget", "credit_budget", "novelty_threshold",
                     "expansion_policy", "stream"):
            assert name in params, f"run() must accept {name}"
        assert params["novelty_threshold"].default == 0.30
        assert params["expansion_policy"].default == "best_first"


# ---------------------------------------------------------------------------
# (a) best-first frontier pops in priority descending order
# ---------------------------------------------------------------------------

class TestFrontier:
    def _node(self, tr, i, priority):
        n = tr.ResearchNode(id=f"0.{i}", question=f"q{i}", parent_id="0", depth=1)
        n.priority = priority
        return n

    def test_pop_descending_priority(self):
        tr = _tr()
        f = tr.Frontier()
        for i, p in enumerate([0.35, 0.9, 0.1, 0.62]):
            f.push(self._node(tr, i, p))
        popped = [f.pop().priority for _ in range(4)]
        assert popped == [0.9, 0.62, 0.35, 0.1]
        assert len(f) == 0

    def test_order_independent_of_insertion(self):
        tr = _tr()
        f = tr.Frontier()
        for i, p in enumerate([0.1, 0.2, 0.3, 0.4]):  # ascending insert
            f.push(self._node(tr, i, p))
        assert [f.pop().priority for _ in range(4)] == [0.4, 0.3, 0.2, 0.1]


# ---------------------------------------------------------------------------
# (b) novelty < threshold -> PRUNED, never expanded (== threshold survives)
# ---------------------------------------------------------------------------

class TestNoveltyPrune:
    def test_low_novelty_child_pruned_and_never_expanded(self):
        children = {ROOT_Q: [LOW_Q, C0_Q]}
        novelty = {ROOT_Q: 1.0, LOW_Q: 0.12, C0_Q: 0.30}
        result, _, rec = _run_tree(children, novelty, {}, max_depth=2,
                                   max_nodes=10, novelty_threshold=0.30)
        assert ROOT_Q in rec["expanded"], "root must be expanded"
        low = _by_question(result, LOW_Q)
        assert low["status"] == "pruned"
        assert low["children"] == []
        assert LOW_Q not in rec["expanded"], "a pruned node must never be expanded"
        # boundary: novelty == threshold is NOT pruned
        assert _by_question(result, C0_Q)["status"] != "pruned"
        root = _by_question(result, ROOT_Q)
        assert root["parent_id"] is None
        assert root["depth"] == 0


# ---------------------------------------------------------------------------
# (c) question-embedding dedup: cosine >= 0.92 dropped, < 0.92 kept
# ---------------------------------------------------------------------------

class TestEmbeddingDedup:
    def test_near_duplicate_question_dropped(self):
        children = {ROOT_Q: [DUP_A_Q, DUP_B_Q, NEAR_Q, DIST_Q]}
        result, _, _ = _run_tree(children, {}, {}, max_depth=2, max_nodes=20)
        qs = _questions(result)
        assert DUP_A_Q in qs, "first candidate must be kept"
        assert DUP_B_Q not in qs, "cosine 0.95 >= 0.92 against DUP_A must be dropped"
        assert NEAR_Q in qs, "cosine 0.90 < 0.92 must be kept"
        assert DIST_Q in qs
        root = _by_question(result, ROOT_Q)
        assert len(root["children"]) == 3, "root must have exactly the 3 deduped children"


# ---------------------------------------------------------------------------
# (d) post-order synthesis leaf->root; citation map = stable-id URL union
# ---------------------------------------------------------------------------

_D_CHILDREN = {ROOT_Q: [C0_Q, C1_Q], C0_Q: [C00_Q]}
_D_SOURCES = {ROOT_Q: [URL_A], C0_Q: [URL_B, URL_A], C00_Q: [URL_C], C1_Q: [URL_B]}


class TestPostOrderSynthesis:
    def test_rollup_is_post_order_root_last(self):
        result, _, rec = _run_tree(_D_CHILDREN, {}, _D_SOURCES,
                                   max_depth=3, max_nodes=10)
        nodes = _nodes(result)
        assert _questions(result) == {ROOT_Q, C0_Q, C1_Q, C00_Q}
        order = rec["synth"]
        assert sorted(order) == sorted(nodes.keys()), (
            "synthesize_node must run exactly once per non-pruned node"
        )
        for nid, n in nodes.items():
            for child_id in n["children"]:
                assert order.index(child_id) < order.index(nid), (
                    f"child {child_id} must be synthesized before parent {nid}"
                )
        root_id = next(nid for nid, n in nodes.items() if n["parent_id"] is None)
        assert order[-1] == root_id, "root synthesis must come last"
        assert f"[synthesis:{ROOT_Q}]" in result["report_md"], (
            "the final report must be built from the root roll-up"
        )

    def test_citation_map_is_stable_url_union(self):
        r1, _, _ = _run_tree(_D_CHILDREN, {}, _D_SOURCES, max_depth=3, max_nodes=10)
        r2, _, _ = _run_tree(_D_CHILDREN, {}, _D_SOURCES, max_depth=3, max_nodes=10)
        cm = r1["citation_map"]
        assert isinstance(cm, dict)
        assert set(cm.values()) == {URL_A, URL_B, URL_C}, (
            "citation map must be the URL union across the whole tree"
        )
        assert len(cm) == 3, "each URL must appear exactly once"
        assert all(isinstance(k, str) for k in cm)
        assert cm == r2["citation_map"], "citation ids must be stable across runs"


# ---------------------------------------------------------------------------
# (e) budget exhaustion -> remaining nodes PENDING, synthesis still produced
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    def test_pending_reported_and_synthesis_still_produced(self):
        children = {ROOT_Q: [C0_Q, C1_Q, DIST_Q]}
        result, _, _ = _run_tree(children, {}, _D_SOURCES,
                                 max_depth=2, max_nodes=2)
        assert _questions(result) == {ROOT_Q, C0_Q, C1_Q, DIST_Q}, (
            "accepted children must exist in the tree even when unresearched"
        )
        statuses = [n["status"] for n in _nodes(result).values()]
        assert "pending" in statuses, (
            "nodes beyond the budget must be reported as pending"
        )
        assert result["report_md"].strip(), (
            "a synthesis must still be produced on budget exhaustion"
        )


# ---------------------------------------------------------------------------
# (f) wall-clock budget: expansion stops in time for the roll-up to still run
# ---------------------------------------------------------------------------

_F_CHILDREN = {ROOT_Q: [C0_Q, C1_Q], C0_Q: [C00_Q]}


class TestTimeBudget:
    def test_wall_clock_budget_cuts_expansion_and_still_synthesizes(self):
        # 400 fake seconds per node vs a 900s budget: root, C0, C1 get researched,
        # then 1200 >= 900 stops the loop with C00 still on the frontier
        result, _, _ = _run_tree(_F_CHILDREN, {}, _D_SOURCES, seconds_per_node=400.0,
                                 max_depth=3, max_nodes=40, time_budget_s=900.0)
        stats = result["stats"]
        assert stats["researched"] == 3, "the 4th node must not start past the budget"
        assert stats["time_budget_exhausted"] is True
        assert stats["pending_count"] == 1, "the un-popped node stays on the frontier"
        assert _by_question(result, C00_Q)["status"] == "pending"
        assert result["report_md"].strip(), (
            "a synthesis must still be produced when the clock runs out"
        )
        assert f"[synthesis:{ROOT_Q}]" in result["report_md"]

    def test_ample_budget_researches_whole_tree(self):
        result, _, _ = _run_tree(_F_CHILDREN, {}, _D_SOURCES, seconds_per_node=400.0,
                                 max_depth=3, max_nodes=40, time_budget_s=10_000.0)
        stats = result["stats"]
        assert stats["researched"] == 4
        assert stats["time_budget_exhausted"] is False
        assert stats["pending_count"] == 0
        assert "pending" not in [n["status"] for n in _nodes(result).values()]


# ---------------------------------------------------------------------------
# research_node reuses GPTResearcher (mocked — no network anywhere)
# ---------------------------------------------------------------------------

class TestResearchNodeWiring:
    def test_research_node_uses_mocked_gpt_researcher(self):
        tr = _tr()
        inst = mock.MagicMock()
        inst.conduct_research = mock.AsyncMock(return_value=["context"])
        inst.write_report = mock.AsyncMock(return_value="node report body")
        inst.visited_urls = {URL_A}
        inst.get_costs.return_value = 0.0
        cls = mock.MagicMock(return_value=inst)
        llm = mock.AsyncMock(return_value="digest text")

        skill = tr.TreeResearchSkill(_FakeResearcher(Config()))
        node = tr.ResearchNode(id="0", question=ROOT_Q, parent_id=None, depth=0)
        with mock.patch(
            "gpt_researcher.skills.tree_research.GPTResearcher", new=cls
        ), mock.patch(
            "gpt_researcher.skills.tree_research.create_chat_completion", new=llm
        ):
            asyncio.run(skill.research_node(node))

        assert cls.call_count == 1, "each node runs exactly one GPTResearcher"
        assert cls.call_args.kwargs.get("query") == ROOT_Q
        inst.conduct_research.assert_awaited()
        assert node.status == tr.NodeStatus.ANSWERED
        assert URL_A in node.sources
