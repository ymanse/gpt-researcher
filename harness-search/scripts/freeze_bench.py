"""Write bench/manifest.sha256 freezing bench/golden/* + baseline_firecrawl.json.

    python scripts/freeze_bench.py

Run ONCE by the s0 agent right before the [sq][s0] commit (the manifest is committed with
the goldens). Idempotent: same files -> identical manifest bytes. Refuses to freeze an
empty/incomplete bench (fail-closed).
"""
from __future__ import annotations

import hconf


def main() -> int:
    goldens = hconf.golden_files()
    if len(goldens) < 5:
        print(f"freeze_refused: only {len(goldens)} golden files (need >=5)")
        return 1
    if not hconf.BASELINE.exists():
        print("freeze_refused: bench/baseline_firecrawl.json missing")
        return 1
    scorer = hconf.BENCH / "score_report.py"
    if not scorer.exists():
        print("freeze_refused: bench/score_report.py missing")
        return 1
    lines = [f"{hconf.sha256(p)}  bench/golden/{p.name}" for p in goldens]
    lines.append(f"{hconf.sha256(hconf.BASELINE)}  bench/baseline_firecrawl.json")
    # The scorer is the measuring instrument: freezing the goldens while leaving it
    # editable locks the window and leaves the door open. Every later score (and the
    # frozen baseline itself) was produced by THESE bytes.
    lines.append(f"{hconf.sha256(scorer)}  bench/score_report.py")
    hconf.MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"froze {len(goldens)} goldens + baseline into bench/manifest.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
