"""Stage 4 RED tests — academic lane promoted to a Firecrawl research adapter.

Target: a FirecrawlResearchSearch adapter (gpt_researcher.retrievers.
firecrawl_research) that runs Firecrawl research paper search plus
related-papers mode="citers" expansion, with from/to recency window
filtering, and is routed by SmartRetriever's "academic" category.

Contract pinned here (GREEN must satisfy exactly this):
- FirecrawlResearchSearch(query, from_date=None, to_date=None, **kwargs)
  .search(max_results=...) -> list of paper dicts each carrying non-empty
  "id", "title", "date" (ISO YYYY-MM-DD).
- Paper search: requests.post to a firecrawl research endpoint; response
  {"success": True, "data": [paper, ...]}.
- Citers expansion: for each seed paper a requests.post with payload
  {"id": <seed id>, "mode": "citers"}; returned papers are merged into the
  result set (they cite the seeds, so they are NEWER than the seeds).
- from/to window: papers dated outside [from_date, to_date] are excluded
  from the final result set.
- SmartRetriever ROUTING_TABLE["academic"] includes "firecrawl_research"
  and get_retriever("firecrawl_research") resolves the adapter class.

Deterministic, no network: requests.post is mocked. Imports happen INSIDE
tests so a missing implementation is a test FAILURE, not a collection error
(RED gate requires errors=0, failed>=1).
"""
from unittest import mock

import pytest

SEED_PAPERS = [
    {
        "id": "arxiv:2501.00001",
        "title": "Seed Paper One",
        "date": "2025-01-10",
        "url": "https://arxiv.org/abs/2501.00001",
    },
    {
        "id": "arxiv:2503.00002",
        "title": "Seed Paper Two",
        "date": "2025-03-05",
        "url": "https://arxiv.org/abs/2503.00002",
    },
    {
        "id": "arxiv:2406.00003",
        "title": "Old Out-of-Window Paper",
        "date": "2024-06-01",
        "url": "https://arxiv.org/abs/2406.00003",
    },
]

# Papers citing each seed — strictly NEWER than every seed date.
CITERS = {
    "arxiv:2501.00001": [
        {
            "id": "arxiv:2606.00010",
            "title": "Newer Citing Paper",
            "date": "2026-06-01",
            "url": "https://arxiv.org/abs/2606.00010",
        },
    ],
}


def _fake_post(url=None, *args, **kwargs):
    """requests.post double for the firecrawl research API.

    Routes on the payload: a related-papers call carries "mode"; anything
    else is treated as paper search.
    """
    payload = kwargs.get("json") or {}
    resp = mock.MagicMock()
    resp.raise_for_status.return_value = None
    resp.status_code = 200
    if payload.get("mode"):
        assert payload["mode"] == "citers", (
            f"related-papers must use mode='citers', got {payload['mode']!r}"
        )
        resp.json.return_value = {
            "success": True,
            "data": CITERS.get(payload.get("id", ""), []),
        }
    else:
        resp.json.return_value = {"success": True, "data": list(SEED_PAPERS)}
    return resp


@pytest.fixture(autouse=True)
def firecrawl_key(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")


def _import_adapter():
    try:
        from gpt_researcher.retrievers.firecrawl_research import FirecrawlResearchSearch
    except ImportError as e:
        pytest.fail(f"FirecrawlResearchSearch not implemented yet: {e}")
    return FirecrawlResearchSearch


def _search(query="graph neural networks", **adapter_kwargs):
    FirecrawlResearchSearch = _import_adapter()
    with mock.patch("requests.post", side_effect=_fake_post) as post:
        papers = FirecrawlResearchSearch(query, **adapter_kwargs).search(max_results=10)
    return papers, post


# ---------------------------------------------------------------------------
# (a) adapter returns papers with id, title, date from mocked paper search
# ---------------------------------------------------------------------------

class TestPaperSearch:
    def test_papers_have_id_title_date(self):
        papers, post = _search()
        assert post.called
        assert papers, "adapter returned no papers from mocked search"
        for p in papers:
            assert p.get("id"), f"paper missing id: {p}"
            assert p.get("title"), f"paper missing title: {p}"
            assert p.get("date"), f"paper missing date: {p}"

    def test_seed_papers_present(self):
        papers, _ = _search()
        ids = {p["id"] for p in papers}
        assert "arxiv:2501.00001" in ids
        assert "arxiv:2503.00002" in ids


# ---------------------------------------------------------------------------
# (b) citers expansion adds papers NEWER than the seed papers
# ---------------------------------------------------------------------------

class TestCitersExpansion:
    def test_citers_added_and_newer_than_seeds(self):
        papers, _ = _search()
        ids = {p["id"] for p in papers}
        assert "arxiv:2606.00010" in ids, "citers expansion did not add the citing paper"
        max_seed_date = max(s["date"] for s in SEED_PAPERS)
        citer = next(p for p in papers if p["id"] == "arxiv:2606.00010")
        assert citer["date"] > max_seed_date, (
            "citers expansion must add papers newer than every seed"
        )

    def test_related_papers_called_with_citers_mode(self):
        _, post = _search()
        citers_calls = [
            c for c in post.call_args_list
            if (c.kwargs.get("json") or {}).get("mode") == "citers"
        ]
        assert citers_calls, "no related-papers call with mode='citers' was made"
        called_ids = {(c.kwargs.get("json") or {}).get("id") for c in citers_calls}
        assert "arxiv:2501.00001" in called_ids, (
            "citers expansion must query related papers for the seed ids"
        )


# ---------------------------------------------------------------------------
# (c) from/to window filtering excludes out-of-window papers
# ---------------------------------------------------------------------------

class TestRecencyWindow:
    def test_out_of_window_paper_excluded(self):
        papers, _ = _search(from_date="2025-01-01", to_date="2026-12-31")
        ids = {p["id"] for p in papers}
        assert "arxiv:2406.00003" not in ids, (
            "paper dated before from_date must be excluded"
        )
        assert "arxiv:2501.00001" in ids
        assert "arxiv:2503.00002" in ids

    def test_in_window_citer_kept(self):
        papers, _ = _search(from_date="2025-01-01", to_date="2026-12-31")
        ids = {p["id"] for p in papers}
        assert "arxiv:2606.00010" in ids, "in-window citing paper must be kept"

    def test_to_date_excludes_newer_papers(self):
        papers, _ = _search(from_date="2025-01-01", to_date="2025-12-31")
        ids = {p["id"] for p in papers}
        assert "arxiv:2606.00010" not in ids, (
            "paper dated after to_date must be excluded"
        )
        assert "arxiv:2503.00002" in ids


# ---------------------------------------------------------------------------
# (d) SmartRetriever academic routing includes the new adapter
# ---------------------------------------------------------------------------

class TestAcademicRouting:
    def test_routing_table_academic_includes_firecrawl_research(self):
        from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE
        names = [entry[0] for entry in ROUTING_TABLE["academic"]]
        assert "firecrawl_research" in names, (
            f"academic lane must route through firecrawl_research, got {names}"
        )

    def test_get_retriever_resolves_firecrawl_research(self):
        from gpt_researcher.actions.retriever import get_retriever
        cls = get_retriever("firecrawl_research")
        assert cls is not None, "get_retriever('firecrawl_research') must resolve"
        assert cls.__name__ == "FirecrawlResearchSearch"
