"""Print gpt-researcher HEAD sha — run IN-GATE by review_common.lua to prove the
review.json was written against the CURRENT code, not recycled from an earlier round."""
from __future__ import annotations

import hconf

if __name__ == "__main__":
    print(hconf.git(hconf.REPO, "rev-parse", "HEAD"))
