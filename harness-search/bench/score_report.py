"""Deterministic S1-S6 scorer for the search-quality bench (spec/search-quality.md).

    python bench/score_report.py --golden bench/golden/<id>.json --report <report.md> \
        [--tree <tree.json>] [--sources <sources.json>] --out <scores.json> \
        [--fetch-cache no_read/fetch_cache]

No model is ever consulted (llm_calls is always 0). The only network access is the S1
citation re-fetch, cached under --fetch-cache by url-hash so re-scoring is idempotent.
Without --tree (baseline reports), citations + the S6 correspondence corpus come from
the report's sibling <name>.sources.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import urllib.request
from urllib.parse import urlparse

CITE = re.compile(r"\[(\d{1,3})\](?!\()")
SIGNUM = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*%|\b\d{4,}\b|\b\d+\.\d+\b")
STOP = {"that", "with", "this", "from", "have", "been", "were", "their", "which",
        "about", "into", "over", "only", "more", "than", "when", "after", "before",
        "while", "where", "also", "each", "other", "them", "they", "these", "those",
        "such", "some", "most", "many", "very", "will", "would", "could", "should",
        "there", "then", "what", "your", "does", "using", "used", "between"}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) sq-bench-scorer/1.0"


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def num_key(s: str) -> str:
    return s.replace(",", "").replace("%", "").strip()


def ctx_tokens(sentence: str) -> set[str]:
    return {t for t in norm(sentence).split()
            if len(t) >= 4 and not t.isdigit() and t not in STOP}


def fetch_text(url: str, cache: pathlib.Path) -> str:
    """Page text for anchor matching; cached (successes AND failures) for idempotence."""
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")
    if key.exists():
        try:
            return json.loads(key.read_text(encoding="utf-8")).get("text", "")
        except json.JSONDecodeError:
            pass
    text = ""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, timeout=30).read(5_000_000)
        raw = body.decode("utf-8", "replace")
        # keep script bodies: SSR frameworks embed the article text in JSON blobs
        text = re.sub(r"<[^>]+>", " ", raw)
    except Exception:  # noqa: BLE001 — any fetch failure means "not grounded", cached as such
        text = ""
    key.write_text(json.dumps({"url": url, "text": text}), encoding="utf-8")
    return text


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]


def domain_hit(domain: str, urls: list[str]) -> bool:
    d = domain.lower()
    return any(domain_of(u) == d or domain_of(u).endswith("." + d) for u in urls)


def claim_sentences(report: str) -> list[str]:
    body = re.sub(r"```.*?```", " ", report, flags=re.DOTALL)
    body = re.sub(r"\[\d{1,3}\]", " ", body)
    body = re.sub(r"\(https?://\S+\)", " ", body)
    body = re.sub(r"https?://\S+", " ", body)
    sents = re.split(r"(?<=[.!?])\s+|\n+", body)
    return [s.strip() for s in sents if s.strip() and SIGNUM.search(s)]


def score_s1(report: str, cites: dict[str, str], cache: pathlib.Path) -> tuple[float, int, int, list[str]]:
    ids = sorted({m.group(1) for m in CITE.finditer(report)}, key=int)
    if not ids:
        return 0.0, 0, 0, []
    uncited = [i for i in ids if i not in cites]
    grounded = 0
    page_cache: dict[str, str] = {}
    for cid in ids:
        if cid not in cites:
            continue
        url = cites[cid]
        if url not in page_cache:
            page_cache[url] = norm(fetch_text(url, cache))
        page = page_cache[url]
        if not page:
            continue
        ok = False
        for m in CITE.finditer(report):
            if m.group(1) != cid:
                continue
            toks = norm(report[max(0, m.start() - 240):m.start()]).split()[-20:]
            for i in range(len(toks) - 2):
                win = toks[i:i + 3]
                if max(len(t) for t in win) >= 4 and " ".join(win) in page:
                    ok = True
                    break
            if ok:
                break
        grounded += 1 if ok else 0
    return grounded / len(ids), len(ids), grounded, uncited


def score_s6(report: str, golden: dict, corpus: list[str]) -> tuple[float, int, int, int]:
    contested = golden["contested"]
    ok = sum(1 for c in contested
             if sum(1 for v in c["values"] if re.search(v, report)) >= 2)
    base = ok / len(contested)

    prepared = []
    for text in corpus:
        if text:
            prepared.append(({num_key(m.group(0)) for m in SIGNUM.finditer(text)},
                             ctx_tokens(text)))
    contradictions = unsupported = 0
    claims = claim_sentences(report)
    for sent in claims:
        nums = {num_key(m.group(0)) for m in SIGNUM.finditer(sent)}
        ctx = ctx_tokens(sent)
        need = min(2, len(ctx))
        if any((nums & cn) and len(ctx & ct) >= need for cn, ct in prepared):
            continue
        if any(cn and len(ctx & ct) >= 3 and not (nums & cn) for cn, ct in prepared):
            contradictions += 1
        else:
            unsupported += 1
    penalty = min(1.0, (contradictions + unsupported) / max(1, len(claims)))
    return base * (1.0 - penalty), ok, contradictions, unsupported


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tree", default=None)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fetch-cache", default="no_read/fetch_cache")
    a = ap.parse_args()

    golden = json.loads(pathlib.Path(a.golden).read_text(encoding="utf-8"))
    report = pathlib.Path(a.report).read_text(encoding="utf-8")
    cache = pathlib.Path(a.fetch_cache)

    cites: dict[str, str] = {}
    corpus: list[str] = []
    url_pool: list[str] = []
    if a.tree:
        tree = json.loads(pathlib.Path(a.tree).read_text(encoding="utf-8"))
        cites = {str(k): v for k, v in tree.get("citations", {}).items()}
        for node in tree.get("nodes", []):
            if node.get("answer"):
                corpus.append(str(node["answer"]))
            url_pool.extend(u for u in node.get("sources", []) if isinstance(u, str))
    else:
        spath = pathlib.Path(a.sources) if a.sources else \
            pathlib.Path(a.report[:-3] + ".sources.json") if a.report.endswith(".md") else None
        if spath and spath.exists():
            src = json.loads(spath.read_text(encoding="utf-8"))
            entries = src.get("sources", src) if isinstance(src, dict) else src
            cites = {str(k): v for k, v in src.get("citations", {}).items()} \
                if isinstance(src, dict) else {}
            for i, e in enumerate(entries if isinstance(entries, list) else []):
                url = e.get("url", "") if isinstance(e, dict) else str(e)
                if url:
                    url_pool.append(url)
                    cites.setdefault(str(i + 1), url)
                if isinstance(e, dict) and e.get("snippet"):
                    corpus.append(str(e["snippet"]))
    url_pool.extend(cites.values())

    s1, cit_total, cit_grounded, uncited = score_s1(report, cites, cache)
    facts = golden["facts"]
    facts_matched = sum(1 for f in facts if re.search(f["pattern"], report))
    s2 = facts_matched / len(facts)
    traps_hit = sum(1 for t in golden["traps"] if re.search(t["pattern"], report))
    s3 = traps_hit / len(golden["traps"])
    domains = golden["required_primary_domains"]
    domains_hit = sum(1 for d in domains if domain_hit(d, url_pool))
    s4 = domains_hit / len(domains)
    areas_hit = sum(1 for c in golden["coverage_areas"] if re.search(c["pattern"], report))
    s5 = areas_hit / len(golden["coverage_areas"])
    s6, contested_ok, contradictions, unsupported = score_s6(report, golden, corpus)

    out = {"qid": golden["id"], "llm_calls": 0,
           "citations_total": cit_total, "citations_grounded": cit_grounded,
           "uncited_ids": uncited, "facts_matched": facts_matched,
           "facts_total": len(facts), "traps_hit": traps_hit,
           "domains_hit": domains_hit, "areas_hit": areas_hit,
           "contested_ok": contested_ok, "contradictions": contradictions,
           "unsupported_claims": unsupported}
    for name, val in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S6", s6)):
        out[name] = round(val, 4)
        out[name + "_pct"] = int(val * 100 + 0.5)
    op = pathlib.Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(" ".join(f"{k}={out[k + '_pct']}" for k in ("S1", "S2", "S3", "S4", "S5", "S6")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
