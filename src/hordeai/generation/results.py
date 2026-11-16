"""Completed generation outputs with matching worker and request details."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult[T]:
    """Items and JSON details have identical ordering and cardinality."""

    items: list[T]
    details: list[str]
