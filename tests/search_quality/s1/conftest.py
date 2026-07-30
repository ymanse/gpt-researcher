"""SmartRetriever retires a failing retriever in module-level process state
(_DEAD_RETRIEVERS, and _TIMED_OUT_ONCE which promotes a repeat timeout into it),
so one test's dead tavily would otherwise change how the next test's routing
behaves. Reset both around every s1 test."""
import pytest

from gpt_researcher.retrievers.smart import smart_retriever as smart_mod


@pytest.fixture(autouse=True)
def _reset_retired_retrievers():
    smart_mod._DEAD_RETRIEVERS.clear()
    smart_mod._TIMED_OUT_ONCE.clear()
    yield
    smart_mod._DEAD_RETRIEVERS.clear()
    smart_mod._TIMED_OUT_ONCE.clear()
