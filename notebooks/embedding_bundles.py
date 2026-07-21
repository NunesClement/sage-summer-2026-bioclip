"""Portable image-embedding artifacts and shared few-shot evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


BUNDLE_SCHEMA_VERSION = 1


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(values, dtype=np.str_)


def ids_digest(ids: Sequence[str]) -> str:
    """Hash an ordered identifier list without relying on NumPy storage details."""
    digest = sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings must not contain zero-length vectors")
    return values / norms


@dataclass(frozen=True)
class EmbeddingBundle:
    """Ordered image identifiers, normalized embeddings, and provenance."""

    ids: np.ndarray
    features: np.ndarray
    manifest: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        ids: Sequence[str],
        features: np.ndarray,
        manifest: Mapping[str, Any],
    ) -> "EmbeddingBundle":
        id_array = _string_array(ids)
        normalized = normalize_features(features)
        if len(id_array) != len(normalized):
            raise ValueError("Identifier and embedding counts differ")

        complete_manifest = dict(manifest)
        complete_manifest.update({
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "ids_sha256": ids_digest(id_array),
            "rows": len(id_array),
            "dimension": normalized.shape[1],
        })
        return cls(id_array, normalized, complete_manifest)

    def require_ids(self, expected_ids: Sequence[str]) -> None:
        expected = _string_array(expected_ids)
        if not np.array_equal(self.ids, expected):
            raise ValueError("Embedding bundle identifiers or order do not match")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            ids=self.ids,
            features=self.features,
            manifest=np.asarray(
                json.dumps(self.manifest, sort_keys=True), dtype=np.str_
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_ids: Sequence[str] | None = None,
    ) -> "EmbeddingBundle":
        with np.load(path, allow_pickle=False) as stored:
            bundle = cls.create(
                stored["ids"],
                stored["features"],
                json.loads(str(stored["manifest"].item())),
            )
        if expected_ids is not None:
            bundle.require_ids(expected_ids)
        return bundle


def producer_manifest(
    *,
    bundle_id: str,
    model_name: str,
    repo_id: str,
    revision: str,
    precision: str,
    evidence_type: str,
    framework: str,
    framework_version: str,
    preprocessing: str,
    quantization: Mapping[str, Any] | None,
    dataset: Mapping[str, Any] | None = None,
    backend: str | None = None,
    export: Mapping[str, Any] | None = None,
    delegation: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the reproducibility record for one embedding producer."""
    return {
        "bundle_id": bundle_id,
        "model": {
            "name": model_name,
            "repo_id": repo_id,
            "revision": revision,
            "tower": "visual",
        },
        "precision": precision,
        "evidence_type": evidence_type,
        "framework": {"name": framework, "version": framework_version},
        "preprocessing": preprocessing,
        "dataset": dataset,
        "quantization": quantization,
        "backend": backend,
        "export": export,
        "delegation": delegation,
        "runtime": runtime,
    }


def nested_support_sets(
    labels: np.ndarray,
    train_indices: np.ndarray,
    class_names: Sequence[str],
    shot_counts: Sequence[int],
    repeat_seed: int,
) -> dict[int, np.ndarray]:
    """Create nested per-class support sets for one repeat."""
    rng = np.random.default_rng(repeat_seed)
    ordered = {
        class_name: rng.permutation(
            train_indices[labels[train_indices] == class_name]
        )
        for class_name in class_names
    }
    return {
        shots: np.concatenate([
            ordered[class_name][:shots] for class_name in class_names
        ])
        for shots in shot_counts
    }


def evaluate_few_shot_bundles(
    bundles: Mapping[str, EmbeddingBundle],
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    class_names: Sequence[str],
    shot_counts: Sequence[int],
    repeat_seeds: Iterable[int],
    estimator_factory: Callable[[], Any],
    metric_functions: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
) -> list[dict[str, Any]]:
    """Refit and score a classifier independently for every embedding bundle."""
    rows: list[dict[str, Any]] = []
    for repeat_seed in repeat_seeds:
        support_sets = nested_support_sets(
            labels,
            train_indices,
            class_names,
            shot_counts,
            repeat_seed,
        )
        for shots, sampled_indices in support_sets.items():
            for variant_name, bundle in bundles.items():
                estimator = estimator_factory()
                estimator.fit(
                    bundle.features[sampled_indices], labels[sampled_indices]
                )
                predictions = estimator.predict(bundle.features[test_indices])
                row = {
                    "variant": variant_name,
                    "model": bundle.manifest["model"]["name"],
                    "precision": bundle.manifest["precision"],
                    "shots_per_species": shots,
                    "repeat": repeat_seed,
                }
                row.update({
                    metric_name: metric(labels[test_indices], predictions)
                    for metric_name, metric in metric_functions.items()
                })
                rows.append(row)
    return rows
