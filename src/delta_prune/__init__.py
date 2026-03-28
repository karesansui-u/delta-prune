"""delta-prune: LLM context contradiction detector and pruner.

Usage:
    from delta_prune import DeltaPrune
    from delta_prune.llm import ClaudeCLI

    prune = DeltaPrune(llm=ClaudeCLI())
    result = prune(messages)
    clean_messages = result.messages
"""

from delta_prune.pruner import DeltaPrune, PruneResult
from delta_prune.extractor import Claim
from delta_prune.resolver import Conflict

__all__ = ["DeltaPrune", "PruneResult", "Claim", "Conflict"]
