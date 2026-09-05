"""Context compression utilities for GPT Researcher.

This module provides classes for compressing and retrieving relevant
context from documents using embeddings and similarity filtering.

The compression pipeline:
1. Splits documents into chunks
2. Filters chunks by embedding similarity to the query
3. Returns the most relevant chunks as context

Classes:
    VectorstoreCompressor: Retrieves context from a vector store.
    ContextCompressor: Compresses raw documents using embedding similarity.
    WrittenContentCompressor: Compresses previously written content sections.
"""

import asyncio
import logging
import os
import re
from typing import Optional

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost
from ..vector_store import VectorStoreWrapper
from .retriever import SearchAPIRetriever, SectionRetriever


def spread_across_sources(docs: list[Document]) -> list[Document]:
    """Round-robin relevance-ordered chunks across the documents they came from.

    EmbeddingsFilter ranks every surviving chunk by similarity alone and the caller
    keeps a flat top-N prefix, so one long page whose wording echoes the query can
    fill the whole retained window while every other page that was scraped
    contributes nothing (defect 2, context starvation: the node then answers from a
    single source and the specifics only the other primary sources carry — a spec
    sentence, a figure from a vendor's own page — never reach the report).
    Interleaving puts every page's best chunk ahead of any page's second, so the
    retained window spans the documents actually read. Order within one source, and
    the relevance order of the sources themselves, are both preserved.
    """
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)
    spread: list[Document] = []
    for rank in range(max((len(chunks) for chunks in by_source.values()), default=0)):
        spread.extend(chunks[rank] for chunks in by_source.values() if rank < len(chunks))
    return spread


logger = logging.getLogger(__name__)

# One pretty_print_docs block: "Source: …\nTitle: …\nContent: …\n".
_CONTEXT_BLOCK = re.compile(r"(?m)^(?=Source: )")

# Bound on the merged context. Above tree_research's own 60k per-node clip on
# purpose, so tree research is unaffected and only the unbounded case — a wide
# sub-query fan on a plain report — is capped.
_MERGE_MAX_CHARS = int(os.environ.get("MERGE_CONTEXT_MAX_CHARS", "120000"))


def merge_sub_query_contexts(contexts: list[str], max_chars: int | None = None) -> str:
    """Merge the per-sub-query contexts into one: drop repeats, interleave, cap.

    The sub-queries of one pass are rephrasings of the same question, so they
    overlap by design and the same page reaches several of them. Chunking is
    deterministic per document, so when two sub-queries retain the same chunk the
    text is byte-identical — ` `.join() paid for it once per sub-query that found
    it, and no downstream stage removes it.

    Concatenation also decides the truncation order: whatever clips the context
    later (tree_research at 60k, a model's window) keeps a prefix, so a long early
    sub-query can consume the whole budget before a later one contributes anything.
    Interleaving is the same rule spread_across_sources applies to chunks, one
    level up: every sub-query's first block comes before any sub-query's second.

    ponytail: exact-duplicate blocks only. Two sources paraphrasing one fact is a
    claim-level judgement — that lives in tree_research's merge, not here.
    """
    if max_chars is None:
        max_chars = _MERGE_MAX_CHARS

    # Non-default prompt families (granite) do not emit "Source:" blocks; there the
    # split yields one block per sub-query and this degrades to today's behaviour
    # plus whole-context dedup.
    groups: list[list[str]] = []
    seen: set[str] = set()
    for context in contexts:
        kept = []
        for block in _CONTEXT_BLOCK.split(context or ""):
            if not block.strip():
                continue
            key = " ".join(block.split())
            if key in seen:
                continue
            seen.add(key)
            kept.append(block.strip("\n"))
        if kept:
            groups.append(kept)

    merged: list[str] = []
    used = 0
    truncated = False
    for rank in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if rank >= len(group):
                continue
            block = group[rank]
            if 0 < max_chars < used + len(block):
                truncated = True
                break
            merged.append(block)
            used += len(block) + 2
        if truncated:
            break

    if truncated:
        logger.warning(
            "Merged context hit MERGE_CONTEXT_MAX_CHARS=%d; kept %d of %d blocks",
            max_chars, len(merged), sum(len(group) for group in groups))
    return "\n\n".join(merged)


class VectorstoreCompressor:
    """Retrieves and compresses context from a vector store.

    Uses similarity search on an existing vector store to find
    relevant documents for a given query.

    Attributes:
        vector_store: The vector store wrapper to search.
        max_results: Maximum number of results to return.
        filter: Optional filter for vector store queries.
    """

    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: Optional[dict] = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the VectorstoreCompressor.

        Args:
            vector_store: The vector store to search.
            max_results: Maximum number of results to return.
            filter: Optional filter dictionary for queries.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter
        self.kwargs = kwargs
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from the vector store.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted string of relevant document content.
        """
        results = await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)


class ContextCompressor:
    """Compresses raw documents to extract relevant context.

    Uses embedding similarity to filter document chunks and return
    only the most relevant content for a given query.

    Attributes:
        documents: List of documents to compress.
        embeddings: Embedding model for similarity calculation.
        max_results: Maximum number of results to return.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the ContextCompressor.

        Args:
            documents: List of documents to compress.
            embeddings: Embedding model instance.
            max_results: Maximum number of results to return.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = os.environ.get("SIMILARITY_THRESHOLD", 0.35)
        self.prompt_family = prompt_family

    def __get_contextual_retriever(self):
        """Build the contextual compression retriever pipeline.

        Returns:
            A ContextualCompressionRetriever configured with text splitting
            and embedding-based filtering.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(
            pages=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        """Get relevant context from documents asynchronously.

        Optimization: Skip expensive compression pipeline for small document sets.
        When documents are already concise, directly use them without embedding-based filtering.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            Formatted string of relevant document content.
        """
        # Optimization: Calculate total content size
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        # If total content is small, skip expensive compression and return directly
        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            # Fast path: no compression needed
            # Same metadata shape SearchAPIRetriever builds on the standard path.
            # Passing the raw page dict through instead left pretty_print_docs
            # reading metadata["source"]/["title"], which a scraped page spells
            # "url"/"title" — so every block on this path printed "Source: None"
            # and the whole pass collapsed to one indistinguishable source.
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', ''),
                    metadata={
                        "title": doc.get("title", ""),
                        "source": doc.get("url", ""),
                    },
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        # Standard path: use compression for large content
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.prompt_family.pretty_print_docs(
            spread_across_sources(relevant_docs), max_results)


class WrittenContentCompressor:
    """Compresses previously written content sections.

    Specialized compressor for finding relevant sections from
    previously written report content, preserving section titles
    and structure.

    Attributes:
        documents: List of written content sections.
        embeddings: Embedding model for similarity calculation.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(self, documents, embeddings, similarity_threshold: float, **kwargs):
        """Initialize the WrittenContentCompressor.

        Args:
            documents: List of written content sections.
            embeddings: Embedding model instance.
            similarity_threshold: Minimum similarity score for inclusion.
            **kwargs: Additional keyword arguments.
        """
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def __get_contextual_retriever(self):
        """Build the contextual compression retriever for sections.

        Returns:
            A ContextualCompressionRetriever configured for section retrieval.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SectionRetriever(
            sections=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    def __pretty_docs_list(self, docs, top_n: int) -> list[str]:
        """Format documents as a list of title/content strings.

        Args:
            docs: List of documents to format.
            top_n: Maximum number of documents to include.

        Returns:
            List of formatted document strings.
        """
        return [f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n" for i, d in enumerate(docs) if i < top_n]

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> list[str]:
        """Get relevant written content sections asynchronously.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            List of formatted section strings.
        """
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.__pretty_docs_list(relevant_docs, max_results)
