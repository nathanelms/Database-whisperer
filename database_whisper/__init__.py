"""database-whisper: Structural profiling for any dataset.

Point it at your data. It tells you what you have.

    import database_whisper as dw

    # Profile any CSV — the one-liner
    report = dw.profile("my_data.csv")
    print(report)

    # Batch router
    router = dw.Router()
    router.ingest(records, identity_fields=["gene", "disease"])
    result = router.query({"gene": "BRAF"}, ask_field="therapy")

    # Streaming / incremental
    live = dw.LiveRouter(identity_fields=["gene", "disease"])
    for record in stream:
        event = live.insert(record)

    # Memory with sleep consolidation
    mem = dw.Memory(identity_fields=["gene", "disease"])
    for fact in facts:
        mem.insert(fact)
"""

__version__ = "0.1.0"

from .profiler import profile, profile_records
from .router import Router
from .ladder import LiveRouter, Memory
from .loader import load, auto_detect_identity, auto_detect_provenance
from .text import (
    auto_detect_concepts,
    extract_concept_instances,
    meaning_addresses,
    resolution_report,
)
from ._types import (
    RouteResult,
    LadderRung,
    InsertEvent,
    SleepEvent,
    StructuralProfile,
)

__all__ = [
    "profile",
    "profile_records",
    "Router",
    "LiveRouter",
    "Memory",
    "load",
    "auto_detect_identity",
    "auto_detect_provenance",
    # Text featurizer
    "auto_detect_concepts",
    "extract_concept_instances",
    "meaning_addresses",
    "resolution_report",
    # Types
    "RouteResult",
    "LadderRung",
    "InsertEvent",
    "SleepEvent",
    "StructuralProfile",
]
