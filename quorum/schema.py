"""Data model for Quorum items and model predictions.

The single design rule: an Item MUST carry at least two independent expert
labels. Single-gold items are refused at construction, because the whole point
of the benchmark is that the reliability of the ground truth is measured, not
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MIN_RATERS = 2


@dataclass(frozen=True)
class Item:
    """One evaluation item with a quorum of expert labels.

    Attributes:
        item_id: Unique identifier.
        task: Task family name (e.g. "sharia_compliance", "credit_grade").
        inputs: Whatever the model under test sees. Opaque to the harness.
        labels: One binary label (0/1) per expert. None marks an expert who did
            not label this item.
        rater_ids: Names of the experts, aligned with labels.
        metadata: Free-form context kept alongside the item.
    """

    item_id: str
    task: str
    inputs: dict[str, Any]
    labels: tuple[int | None, ...]
    rater_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.rater_ids):
            raise ValueError(f"{self.item_id}: labels and rater_ids must align.")
        present = [l for l in self.labels if l is not None]
        if len(present) < MIN_RATERS:
            raise ValueError(
                f"{self.item_id}: Quorum items need at least {MIN_RATERS} expert labels; "
                f"got {len(present)}. Single-gold items are not admitted."
            )
        for l in present:
            if l not in (0, 1):
                raise ValueError(f"{self.item_id}: labels must be 0, 1 or None; got {l!r}.")

    @property
    def present_labels(self) -> list[int]:
        return [l for l in self.labels if l is not None]

    @property
    def n_raters(self) -> int:
        return len(self.present_labels)

    @property
    def positives(self) -> int:
        return sum(self.present_labels)

    @property
    def agreement(self) -> str:
        """'unanimous', 'majority', or 'split' (an even tie)."""
        n, p = self.n_raters, self.positives
        if p == 0 or p == n:
            return "unanimous"
        if n % 2 == 0 and p * 2 == n:
            return "split"
        return "majority"

    @property
    def majority_label(self) -> int | None:
        """The quorum's verdict, or None on a split."""
        if self.agreement == "split":
            return None
        return 1 if self.positives * 2 > self.n_raters else 0


@dataclass(frozen=True)
class Prediction:
    """A model's answer for one item.

    Attributes:
        item_id: Which item.
        label: 0, 1, or None to abstain.
        confidence: Optional probability the model assigns to label 1, in [0, 1].
    """

    item_id: str
    label: int | None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.label not in (0, 1, None):
            raise ValueError(f"{self.item_id}: prediction label must be 0, 1 or None.")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"{self.item_id}: confidence must lie in [0, 1].")
