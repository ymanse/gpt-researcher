"""SmartRetriever retires a failing retriever in module-level process state
(_DEAD_RETRIEVERS), so one test's dead tavily would otherwise change how the next
test's routing behaves. Reset it around every s1 test."""
import pytest

from gpt_researcher.retrievers.smart import smart_retriever as smart_mod


@pytest.fixture(autouse=True)
def _reset_retired_retrievers():
    smart_mod._DEAD_RETRIEVERS.clear()
    yield
    smart_mod._DEAD_RETRIEVERS.clear()
