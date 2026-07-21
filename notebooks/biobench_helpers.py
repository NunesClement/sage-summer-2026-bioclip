"""Data and embedding utilities for the notebook's NeWT benchmark."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tarfile

import numpy as np
import polars as pl
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm

from embedding_bundles import EmbeddingBundle


def _download_with_resume(
    url: str,
    destination: Path,
    chunk_size: int = 8 * 1024**2,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    with requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=60,
    ) as response:
        if downloaded and response.status_code != 206:
            downloaded = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        total = downloaded + int(response.headers.get("content-length", 0))
        mode = "ab" if downloaded else "wb"
        with partial.open(mode) as output, tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.name}")
        for member in tqdm(
            members,
            desc=f"Extracting {archive_path.name}",
            unit="files",
        ):
            try:
                archive.extract(member, destination, filter="data")
            except TypeError:
                archive.extract(member, destination)


def prepare_newt(
    directory: Path,
    *,
    images_url: str,
    labels_url: str,
    expected_images: int,
) -> tuple[Path, Path]:
    """Download and validate the NeWT image and label archives."""
    directory.mkdir(parents=True, exist_ok=True)
    images_directory = directory / "newt2021_images"
    labels_path = directory / "newt2021_labels.csv"

    if not labels_path.exists():
        labels_archive = directory / "newt_labels.tar.gz"
        _download_with_resume(labels_url, labels_archive)
        _safe_extract(labels_archive, directory)
        labels_archive.unlink()

    existing_image_count = (
        sum(1 for _ in images_directory.glob("*.jpg"))
        if images_directory.exists()
        else 0
    )
    if existing_image_count != expected_images:
        images_archive = directory / "newt_images.tar.gz"
        _download_with_resume(images_url, images_archive)
        _safe_extract(images_archive, directory)
        images_archive.unlink()

    image_count = sum(1 for _ in images_directory.glob("*.jpg"))
    if image_count != expected_images:
        raise RuntimeError(
            f"Expected {expected_images:,} NeWT images, "
            f"found {image_count:,}"
        )
    return images_directory, labels_path


def sample_newt_metadata(
    labels_path: Path,
    images_directory: Path,
    *,
    proportion: float,
    seed: int,
    display_names: dict[str, str],
    expected_rows: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[Path]]:
    """Sample each task, split, and label stratum deterministically."""
    full_metadata = (
        pl.read_csv(labels_path)
        .with_columns(
            pl.col("task_cluster").alias("cluster"),
            pl.col("task")
            .replace(display_names)
            .alias("display_name"),
        )
        .sort(["task", "split", "id"])
        .with_row_index("_source_row")
    )
    if full_metadata.height != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {full_metadata.height:,}"
        )

    if proportion == 1:
        metadata = full_metadata
    else:
        rng = np.random.default_rng(seed)
        selected_rows = []
        for stratum in full_metadata.partition_by(
            ["task", "split", "label"],
            maintain_order=True,
        ):
            sample_size = max(
                1,
                int(np.ceil(stratum.height * proportion)),
            )
            selected_rows.extend(
                rng.choice(
                    stratum["_source_row"].to_numpy(),
                    size=sample_size,
                    replace=False,
                ).tolist()
            )
        metadata = (
            full_metadata.filter(
                pl.col("_source_row").is_in(selected_rows)
            )
            .sort(["task", "split", "id"])
        )

    metadata = metadata.drop("_source_row")
    full_metadata = full_metadata.drop("_source_row")
    image_paths = [
        images_directory / f"{image_id}.jpg"
        for image_id in metadata["id"]
    ]
    missing_paths = [path for path in image_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths):,} NeWT images are missing. "
            f"First missing path: {missing_paths[0]}"
        )
    return metadata, full_metadata, image_paths


def encode_image_paths(
    model,
    preprocess,
    image_paths: list[Path],
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> np.ndarray:
    """Create normalized image embeddings with threaded image loading."""

    def load_image(image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return preprocess(image.convert("RGB"))

    batches = []
    encoded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            batch = torch.stack(list(pool.map(load_image, paths)))
            batch = batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            with torch.inference_mode():
                features = model.encode_image(batch, normalize=True)
            batches.append(features.float().cpu())
            encoded += len(batch)
            print(
                f"Created embeddings for {encoded:,}/"
                f"{len(image_paths):,} images",
                end="\r",
            )
    print()
    return torch.cat(batches).numpy()


def load_embedding_cache(
    cache_path: Path,
    expected_ids: np.ndarray,
    manifest: dict,
) -> EmbeddingBundle | None:
    """Load a bundle and migrate the notebook's older feature-only cache."""
    if not cache_path.exists():
        return None
    try:
        return EmbeddingBundle.load(cache_path, expected_ids)
    except KeyError:
        with np.load(cache_path, allow_pickle=False) as cached:
            if not np.array_equal(cached["ids"], expected_ids):
                print(f"Ignoring stale cache at {cache_path}")
                return None
            bundle = EmbeddingBundle.create(
                expected_ids,
                cached["features"],
                manifest,
            )
        bundle.save(cache_path)
        print(f"Added a producer manifest to {cache_path}")
        return bundle
    except (ValueError, TypeError):
        print(f"Ignoring stale cache at {cache_path}")
        return None
