"""Construct the taxonomic text formats used to train BioCLIP models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl


TAXONOMIC_RANKS = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
)

TRAINING_TEMPLATE_NAMES = (
    "com",
    "common_name",
    "sci",
    "sci_com",
    "scientific_name",
    "taxon",
    "taxonTag",
    "taxonTag_com",
    "taxon_com",
    "taxonomic_name",
)


def _clean_term(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _rank_value_for_text(
    rank: str,
    value: Any,
    row: Mapping[str, Any],
) -> str | None:
    value = _clean_term(value)
    if rank == "species" and value:
        genus = _clean_term(row.get("genus"))
        if genus and value.startswith(genus + " "):
            return value[len(genus) + 1 :]
    return value


def _most_specific_linnaean_name(row: Mapping[str, Any]) -> str:
    for rank in reversed(TAXONOMIC_RANKS):
        value = _clean_term(row.get(rank))
        if value:
            return value
    raise ValueError("A candidate class has no Linnaean rank term")


def _taxonomic_name(row: Mapping[str, Any]) -> str:
    terms = [
        _rank_value_for_text(rank, row.get(rank), row)
        for rank in TAXONOMIC_RANKS
    ]
    return " ".join(term for term in terms if term)


def _tagged_lineage(row: Mapping[str, Any]) -> str:
    terms = []
    for rank in TAXONOMIC_RANKS:
        value = _rank_value_for_text(rank, row.get(rank), row)
        if value:
            terms.extend([rank, value])
    return " ".join(terms)


def build_class_definitions(
    sample_metadata: pl.DataFrame,
    group_metadata: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Build one taxonomy and common-name record for each species class."""
    taxonomy_rows = (
        sample_metadata.select(TAXONOMIC_RANKS)
        .unique(subset=["species"])
        .sort("species")
    )

    definitions = []
    for row in taxonomy_rows.iter_rows(named=True):
        scientific_name = _most_specific_linnaean_name(row)
        common_names = (
            group_metadata.filter(pl.col("species") == row["species"])
            .select("common_name")
            .drop_nulls()
            .unique()
            .sort("common_name")["common_name"]
            .to_list()
        )
        common_names = [
            name.strip() for name in common_names if name and name.strip()
        ]
        if not common_names:
            common_names = [scientific_name]

        definitions.append(
            {
                "class_name": row["species"],
                "scientific_name": scientific_name,
                "taxonomic_name": _taxonomic_name(row),
                "tagged_lineage": _tagged_lineage(row),
                "common_names": common_names,
            }
        )
    return definitions


def training_template_prompts(
    class_definition: Mapping[str, Any],
    common_name: str,
) -> list[str]:
    """Return the ten BioCLIP training text formats for one taxon name."""
    scientific_name = class_definition["scientific_name"]
    taxonomic_name = class_definition["taxonomic_name"]
    tagged_lineage = class_definition["tagged_lineage"]
    return [
        f"a photo of {common_name}.",
        common_name,
        f"a photo of {scientific_name}.",
        f"a photo of {scientific_name} with common name {common_name}.",
        scientific_name,
        f"a photo of {taxonomic_name}.",
        f"a photo of {tagged_lineage}.",
        f"a photo of {tagged_lineage} with common name {common_name}.",
        f"a photo of {taxonomic_name} with common name {common_name}.",
        taxonomic_name,
    ]


def build_training_prompt_ensemble(
    class_definitions: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[slice]]:
    """Flatten each class's training prompts and retain its averaging slice."""
    prompts = []
    class_slices = []
    for definition in class_definitions:
        start = len(prompts)
        for common_name in definition["common_names"]:
            prompts.extend(training_template_prompts(definition, common_name))
        class_slices.append(slice(start, len(prompts)))
    return prompts, class_slices
