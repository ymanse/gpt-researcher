"""Frozen-bench check — run IN-GATE by every post-s0 gate (law: golden/baseline read-only).

    python scripts/check_frozen.py

Recomputes sha256 of every bench/golden/*.json and bench/baseline_firecrawl.json and
compares against bench/manifest.sha256 (written once by freeze_bench.py at s0 pass).
Fail-closed: missing manifest, missing files, extra golden files, or any hash drift
all print frozen_ok=0 with a prescriptive reason.
Prints: frozen_ok=1 golden_count=N reason=-   or   frozen_ok=0 ... reason=<why>
"""
from __future__ import annotations

import hconf


def main() -> int:
    if not hconf.MANIFEST.exists():
        print("frozen_ok=0 golden_count=0 reason=manifest_missing_run_freeze_bench_at_s0")
        return 0
    want: dict[str, str] = {}
    for ln in hconf.MANIFEST.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        h, _, rel = ln.partition("  ")
        want[rel.strip()] = h.strip()

    golden = {f"bench/golden/{p.name}" for p in hconf.golden_files()}
    golden_in_manifest = {k for k in want if k.startswith("bench/golden/")}
    if golden != golden_in_manifest:
        extra = sorted(golden - golden_in_manifest)
        missing = sorted(golden_in_manifest - golden)
        print(f"frozen_ok=0 golden_count={len(golden)} "
              f"reason=golden_set_changed_extra:{','.join(extra) or '-'}_missing:{','.join(missing) or '-'}")
        return 0
    if "bench/baseline_firecrawl.json" not in want:
        print("frozen_ok=0 golden_count=0 reason=baseline_not_in_manifest")
        return 0

    for rel, h in sorted(want.items()):
        p = hconf.HARNESS / rel
        if not p.exists():
            print(f"frozen_ok=0 golden_count={len(golden)} reason=frozen_file_deleted:{rel}")
            return 0
        if hconf.sha256(p) != h:
            print(f"frozen_ok=0 golden_count={len(golden)} reason=frozen_file_MODIFIED:{rel}")
            return 0
    print(f"frozen_ok=1 golden_count={len(golden)} reason=-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
