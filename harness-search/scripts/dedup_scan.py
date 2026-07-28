"""Deterministic redundancy scan of a report — the evidence the dedup gate reads.

    python scripts/dedup_scan.py <report.md> [<report.md> ...]

Emits one JSON object per report on stdout plus an aggregate line. NO LLM, no network:
near-duplication is measured as token-set Jaccard overlap between text units, which is
reproducible byte-for-byte from the same report.

What it counts
  sentence unit  a sentence >= MIN_TOKENS content words (headings, bullets markers and
                 citation brackets stripped) — short connective lines are not evidence
                 of redundancy and would only dilute the ratio.
  duplicate      a unit whose token set overlaps an EARLIER unit with Jaccard >= DUP_J.
                 Only the later occurrence is counted, so a fact stated once is never a
                 duplicate no matter how many times it is echoed later.
  block          a paragraph (blank-line separated), same rule at BLOCK_J.

Reported fields (all bound to a positive scanned count so an empty report cannot look
clean — law 1/4):
  sentences_scanned, dup_sentences, dup_sentence_pct
  blocks_scanned, dup_blocks, dup_block_pct
  max_repeat_cluster  largest group of mutually near-identical sentences (a "said the
                      same thing N times" signal that a ratio alone hides)
  headings            markdown headings, a cheap organization signal
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

MIN_TOKENS = 6          # ignore fragments; they are not what "duplicated content" means
DUP_J = 0.80            # sentence-level near-duplicate threshold
BLOCK_J = 0.70          # paragraphs paraphrase more, so a looser bar
_CITE = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(_CITE.sub(" ", text).lower()) if len(w) > 2)


def units(text: str, block: bool) -> list[frozenset[str]]:
    raw = re.split(r"\n\s*\n", text) if block else _SENT.split(text.replace("\n", " "))
    out = []
    for r in raw:
        r = re.sub(r"^[\s>#*\-\d.]+", "", r)
        t = tokens(r)
        if len(t) >= MIN_TOKENS:
            out.append(t)
    return out


def count_dups(us: list[frozenset[str]], thresh: float) -> tuple[int, int]:
    """(duplicates, largest mutually-similar cluster). Candidate generation uses an
    inverted index over each unit's rarest tokens, so this stays linear-ish instead of
    comparing every pair."""
    df: dict[str, int] = {}
    for u in us:
        for t in u:
            df[t] = df.get(t, 0) + 1
    index: dict[str, list[int]] = {}
    dups = 0
    cluster: dict[int, int] = {}
    for i, u in enumerate(us):
        rare = sorted(u, key=lambda t: df[t])[:8]
        seen: set[int] = set()
        for t in rare:
            seen.update(index.get(t, ()))
        hit = -1
        for j in seen:
            v = us[j]
            inter = len(u & v)
            if inter and inter / len(u | v) >= thresh:
                hit = j
                break
        if hit >= 0:
            dups += 1
            root = cluster.get(hit, hit)
            cluster[i] = root
            cluster[hit] = root
        for t in rare:
            index.setdefault(t, []).append(i)
    sizes: dict[int, int] = {}
    for _, root in cluster.items():
        sizes[root] = sizes.get(root, 0) + 1
    return dups, (max(sizes.values()) if sizes else 0)


def scan(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    sents = units(text, block=False)
    blocks = units(text, block=True)
    ds, cluster = count_dups(sents, DUP_J)
    db, _ = count_dups(blocks, BLOCK_J)
    return {
        "report": path.name,
        "sentences_scanned": len(sents),
        "dup_sentences": ds,
        "dup_sentence_pct": round(100 * ds / len(sents)) if sents else 100,
        "blocks_scanned": len(blocks),
        "dup_blocks": db,
        "dup_block_pct": round(100 * db / len(blocks)) if blocks else 100,
        "max_repeat_cluster": cluster,
        "headings": len(re.findall(r"^#{1,6}\s+\S", text, re.M)),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: dedup_scan.py <report.md> [...]")
        return 2
    rows = [scan(pathlib.Path(a)) for a in sys.argv[1:]]
    for r in rows:
        print(json.dumps(r, sort_keys=True))
    if rows:
        print("AGG " + json.dumps({
            "reports": len(rows),
            "sentences_scanned_total": sum(r["sentences_scanned"] for r in rows),
            "dup_sentence_pct_max": max(r["dup_sentence_pct"] for r in rows),
            "dup_block_pct_max": max(r["dup_block_pct"] for r in rows),
            "max_repeat_cluster": max(r["max_repeat_cluster"] for r in rows),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
