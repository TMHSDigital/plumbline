"""plumbline: choose and configure a decision model on your own labeled data.

plumbline is not a leaderboard. JevBench already ranks public systems against
each other. plumbline answers the three questions no public ranking can:
how each candidate behaves on your labeled data, what its reported probability
becomes after recalibration fitted on that data, and what a cascade at the
resulting threshold actually costs you.
"""

from plumbline.types import (
    PROBABILITY_SEMANTICS,
    Case,
    CaseRefusedError,
    PlumblineError,
    Prediction,
    ProbabilitySemantics,
    UnknownAdapterError,
    docs_confidence,
)

__version__ = "0.1.0"

__all__ = [
    "PROBABILITY_SEMANTICS",
    "Case",
    "CaseRefusedError",
    "PlumblineError",
    "Prediction",
    "ProbabilitySemantics",
    "UnknownAdapterError",
    "__version__",
    "docs_confidence",
]
