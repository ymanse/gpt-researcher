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
    lines = [f"{hconf.sha256(p)}  bench/golden/{p.name}" for p in goldens]
    lines.append(f"{hconf.sha256(hconf.BASELINE)}  bench/baseline_firecrawl.json")
    hconf.MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"froze {len(goldens)} goldens + baseline into bench/manifest.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
