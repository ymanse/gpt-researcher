"""s9: a merge measured on a DEAD similarity signal must not read as "no redundancy".

`_unit_vectors` degrades to word-overlap candidate selection whenever the embedding
provider is down, out of quota or unconfigured. That is right for production — a
research run should not crash because an embeddings bill lapsed — and useless for
measurement: the redundancy in this corpus is TOPICAL, not lexical (repeated 5-grams
0-2%), so under word overlap the restated pairs never become candidates, the
equivalence judge is never asked about them, and the merge is a no-op whose numbers are
identical to a tree that genuinely had nothing to merge.

Measured 2026-07-29/30: the OpenAI embeddings quota expired mid-loop (HTTP 429
insufficient_quota). Every later offline round reported `lifted 10/10`, `ratio 115%`,
`S2 = baseline` — the signature of a broken implementation — and three refit rounds plus
three human round-grants were spent on an implementation that was never the defect.

So the counters have to carry the provenance of the signal, not just its results.
`embedded_units < claim_units` is what harness-search/scripts/d1_offline.lua reads to
refuse the measurement outright (a HARD fail, not a refit round — no edit to the merge
can put an embedding provider back).

Reuses the frozen corpus fixture from test_s9_merge_contract rather than restating it:
that file pins the sentences, the seams and the offline model stand-in, and is
RED-recorded by hash, so it is the one definition of what a run over this corpus is.
"""
from __future__ import annotations

from unittest import mock

import gpt_researcher.skills.tree_research as tr
import test_s9_merge_contract as fx


async def test_a_live_embedding_signal_is_reported_as_covering_every_claim_unit():
    _, result = await fx._assemble()
    stats = result["merge_stats"]
    assert stats.get("claim_units", 0) >= 2, (
        "the fixture tree must yield claim units for the merge to consider; "
        f"merge_stats={stats}"
    )
    assert stats.get("embedded_units") == stats["claim_units"], (
        "every claim unit was embedded through the patched seam, so the counters must "
        f"say so — otherwise a healthy run is indistinguishable from a degraded one: {stats}"
    )


async def test_a_dead_embedding_provider_leaves_the_run_marked_unusable():
    """Two outcomes at once, and they pull in opposite directions.

    The report must still be produced (production degrades, it does not crash), AND the
    counters must record that candidate selection never had a real signal — so nothing
    downstream can read the resulting no-op as "this tree had no redundancy".
    """
    skill = fx._tree()

    async def out_of_quota(_text: str) -> list:
        raise RuntimeError("Error code: 429 - {'error': {'type': 'insufficient_quota'}}")

    skill.embed_question = out_of_quota
    with mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(side_effect=fx._model)):
        result = await skill.assemble_report(fx.ROOT_Q)

    assert result["report_md"].strip(), (
        "a dead embedding provider must not cost the report — the fallback exists so a "
        "live research run survives it"
    )
    stats = result["merge_stats"]
    assert stats.get("claim_units", 0) >= 2, f"the merge still ran: {stats}"
    assert stats.get("embedded_units") == 0, (
        "candidates were picked by word overlap, so the similarity signal was absent for "
        f"EVERY unit and the measurement is void, not merely bad: {stats}"
    )
