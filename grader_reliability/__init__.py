"""Dataset-agnostic grader-reliability metrics.

Public API:
    reliability_report      - full picture for one labelled dataset
    krippendorff_alpha      - inter-rater agreement (nominal or ordinal)
    accuracy_by_agreement   - model accuracy split by grader agreement level
    agreement_level         - classify one item's votes (unanimous/majority/tie)
    exact_agreement_rate, binary_contested_rate, single_vote_rate
    binarize
"""

from grader_reliability.reliability import (
    ReliabilityReport,
    accuracy_by_agreement,
    agreement_level,
    binarize,
    binary_contested_rate,
    exact_agreement_rate,
    krippendorff_alpha,
    reliability_report,
    single_vote_rate,
)

__all__ = [
    "ReliabilityReport",
    "accuracy_by_agreement",
    "agreement_level",
    "binarize",
    "binary_contested_rate",
    "exact_agreement_rate",
    "krippendorff_alpha",
    "reliability_report",
    "single_vote_rate",
]
