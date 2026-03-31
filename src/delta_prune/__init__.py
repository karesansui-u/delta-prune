"""delta-prune: LLM context contradiction detector and pruner.

Chat (OpenAI-style messages)::

    from delta_prune import DeltaPrune
    from delta_prune.llm import ClaudeCLI

    prune = DeltaPrune(llm=ClaudeCLI())
    result = prune(messages)
    clean_messages = result.messages

RAG (retrieved text chunks)::

    result = prune.filter_chunks(chunks)
    context = "\n\n".join(result.filtered_chunks)
"""

from delta_prune.pruner import ChunkPruneResult, DeltaPrune, PruneResult
from delta_prune.extractor import Claim
from delta_prune.resolver import Conflict

__all__ = ["DeltaPrune", "PruneResult", "ChunkPruneResult", "Claim", "Conflict"]
