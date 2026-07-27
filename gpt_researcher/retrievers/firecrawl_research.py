# Firecrawl Research Retriever
#
# Academic-lane adapter over Firecrawl's research API: paper search
# (/v2/search/research/papers) plus related-papers expansion with
# mode="citers" (papers citing the seeds are newer than the seeds), with an
# optional from/to recency window applied client-side.
# Requires FIRECRAWL_API_KEY (https://firecrawl.dev).

import logging
import os
import re
from datetime import date, timedelta
from urllib.parse import quote

import requests

_RESEARCH_PAPERS = "https://api.firecrawl.dev/v2/search/research/papers"
_DATE_KEYS = (
    "date", "publishedDate", "published", "publicationDate",
    "created", "createdAt", "updated", "updatedAt",
)
_ARXIV_YYMM = re.compile(r"arxiv:(\d{2})(\d{2})\.")


logger = logging.getLogger(__name__)


class FirecrawlResearchSearch:
    """
    Firecrawl research retriever returning paper dicts each carrying
    non-empty "id", "title", "date" (ISO YYYY-MM-DD), plus "href"/"body"
    for downstream scraping compatibility.
    """

    def __init__(self, query, from_date=None, to_date=None, recency_days=None,
                 query_domains=None, **kwargs):
        self.query = query
        self.from_date = str(from_date)[:10] if from_date else None
        self.to_date = str(to_date)[:10] if to_date else None
        if self.from_date is None and recency_days:
            self.from_date = (date.today() - timedelta(days=int(recency_days))).isoformat()
        self.api_key = os.getenv("FIRECRAWL_API_KEY")

    def search(self, max_results=10):
        print(f"Searching research papers for {self.query}...")
        if not self.api_key:
            print("Firecrawl research: no FIRECRAWL_API_KEY set — skipping.")
            return []
        try:
            return self._search(max_results)
        except Exception as e:
            logger.error(f"Error: {e}. Failed fetching papers from Firecrawl research. Resulting in empty response.")
            return []

    def _search(self, max_results):
        seeds = [p for p in (self._normalize(r) for r in self._search_papers(max_results)) if p]

        merged = {p["id"]: p for p in seeds}
        from_citers = 0
        for seed in seeds[:8]:
            for raw in self._citers_of(seed["id"]):
                p = self._normalize(raw)
                if p and p["id"] not in merged:
                    merged[p["id"]] = p
                    from_citers += 1

        papers = list(merged.values())
        in_window = [p for p in papers if self._in_window(p["date"])]
        print(
            f"TIERA_EVIDENCE stage=4 papers_count={len(papers)} "
            f"papers_in_window={len(in_window)} from_citers_count={from_citers}",
            flush=True,
        )
        return in_window[:max_results]

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _search_papers(self, max_results):
        payload = {"query": self.query, "k": min(max(max_results, 1), 100)}
        if self.from_date:
            payload["from"] = self.from_date
        if self.to_date:
            payload["to"] = self.to_date
        js = self._api(_RESEARCH_PAPERS, payload, get_params=payload)
        return js.get("data") or js.get("results") or []

    def _citers_of(self, paper_id):
        url = f"{_RESEARCH_PAPERS}/{quote(str(paper_id), safe='')}/similar"
        payload = {"id": paper_id, "mode": "citers", "intent": self.query, "k": 10}
        js = self._api(url, payload, get_params={"mode": "citers", "intent": self.query, "k": 10})
        return js.get("data") or js.get("results") or []

    def _api(self, url, payload, get_params=None):
        # ponytail: POST-first is the pinned stage-4 test contract; the public
        # research endpoints are documented as GET, so fall back on 404/405.
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code in (404, 405) and get_params is not None:
            resp = requests.get(url, params=get_params, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Normalization & window
    # ------------------------------------------------------------------

    def _normalize(self, raw):
        pid = raw.get("id") or raw.get("primaryId") or raw.get("paperId")
        title = raw.get("title")
        pdate = self._date_of(raw, pid)
        if not (pid and title and pdate):
            return None
        href = raw.get("url") or raw.get("href") or ""
        if not href and str(pid).startswith("arxiv:"):
            href = f"https://arxiv.org/abs/{str(pid).split(':', 1)[1]}"
        return {
            **raw,
            "id": pid,
            "title": title,
            "date": pdate,
            "href": href,
            "body": raw.get("abstract") or raw.get("body") or title,
        }

    @staticmethod
    def _date_of(raw, pid):
        for k in _DATE_KEYS:
            v = raw.get(k)
            if v:
                return str(v)[:10]
        # ponytail: arxiv:YYMM.NNNNN → first of that month, better than dropping the paper
        m = _ARXIV_YYMM.match(str(pid or ""))
        if m and 1 <= int(m.group(2)) <= 12:
            return f"20{m.group(1)}-{m.group(2)}-01"
        return ""

    def _in_window(self, pdate):
        if self.from_date and pdate < self.from_date:
            return False
        if self.to_date and pdate > self.to_date:
            return False
        return True
