"""Offline re-synthesis: replay one captured tree's roll-up + report assembly for free.

    python scripts/resynth.py --resynth <x.resynth.json> --tree <x.tree.json> \
                              --out <dir> --as <golden_id>

Writes <dir>/<golden_id>.report.md.

A live deep_tree_research costs ~330 Firecrawl credits and ~12 minutes per query, and
none of that buys anything for the roll-up: create_chat_completion appears exactly twice
in gpt_researcher/skills/tree_research.py and both call sites are upstream of the
assembly, which reads only self.nodes and self._read_docs. So once _persist's
<stem>.resynth.json sidecar is on disk the whole synthesis can be replayed offline —
which is what makes iterating on the merge cost minutes instead of hours.

This runner CALLS TreeResearchSkill.assemble_report; it must never re-implement it. A
private copy would reproduce the captured report today and diverge silently the first
time the real assembly is edited — and every later dedup stage measures this runner
instead of a live run, so nobody would see the drift (the d0 gate checks both
shares_assembly and bytes_identical for exactly that reason).

The report is written as BYTES with the string's own newlines: the reference report was
written by _persist inside the Linux container, so translating "\\n" to the host's
os.linesep here would break byte identity on Windows for a difference that is not real.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import types

import hconf

sys.path.insert(0, str(hconf.REPO))

from gpt_researcher.skills.tree_research import (
    NodeStatus,
    ResearchNode,
    TreeResearchSkill,
)


def restore(sidecar: dict) -> TreeResearchSkill:
    """Rebuild the skill state assemble_report reads, in the captured insertion order."""
    # only the __init__ attributes are touched; no researcher call is reachable from
    # the assembly path, so a stub is the honest stand-in for a live GPTResearcher
    stub = types.SimpleNamespace(
        tone=None, websocket=None, headers={}, visited_urls=set(), query="",
        cfg=types.SimpleNamespace(config_path=None),
    )
    skill = TreeResearchSkill(stub)
    for nd in sidecar.get("nodes") or []:
        node = ResearchNode(id=nd["id"], question=nd.get("question", ""),
                            parent_id=nd.get("parent_id"), depth=int(nd.get("depth", 0)))
        node.status = NodeStatus(nd["status"])
        node.answer_md = nd.get("answer_md", "")
        node.answer_digest = nd.get("answer_digest", "")
        node.learnings = list(nd.get("learnings") or [])
        node.sources = list(nd.get("sources") or [])
        node.novelty = float(nd.get("novelty", 1.0))
        node.priority = float(nd.get("priority", 0.0))
        node.children = list(nd.get("children") or [])
        skill.nodes[node.id] = node
    skill._read_docs = dict(sidecar.get("read_docs") or {})
    return skill


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resynth", required=True, help="the <stem>.resynth.json sidecar")
    ap.add_argument("--tree", required=True, help="the matching <stem>.tree.json")
    ap.add_argument("--out", required=True, help="directory to write <id>.report.md into")
    ap.add_argument("--as", dest="gid", required=True, help="golden id / output stem")
    a = ap.parse_args()

    sidecar = json.loads(pathlib.Path(a.resynth).read_text(encoding="utf-8"))
    tree = json.loads(pathlib.Path(a.tree).read_text(encoding="utf-8"))
    # the sidecar owns the query; the tree is the cross-check that the two artifacts
    # describe the same run, and the fallback for a sidecar written without one
    query = sidecar.get("query") or (tree.get("meta") or {}).get("query") or ""

    skill = restore(sidecar)
    if not skill.nodes:
        print(f"resynth_ok=0 reason=sidecar_has_no_nodes:{a.resynth}")
        return 1

    result = asyncio.run(skill.assemble_report(query))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"{a.gid}.report.md"
    report.write_bytes(result["report_md"].encode("utf-8"))
    print(f"resynth_ok=1 id={a.gid} nodes={len(skill.nodes)} "
          f"read_docs={len(skill._read_docs)} chars={len(result['report_md'])} "
          f"citations={len(result['citation_map'])} out={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
