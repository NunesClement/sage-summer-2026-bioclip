"""Runtime and measurement helpers for the notebook's W8A8 experiment."""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import torch

from embedding_bundles import (
    EmbeddingBundle,
    producer_manifest,
)


_pytree_logger = logging.getLogger("torch.utils._pytree")
_pytree_log_level = _pytree_logger.level
_pytree_logger.setLevel(logging.ERROR)
try:
    import torchao
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        quantize_,
    )
    from torchao.utils import (
        get_model_size_in_bytes as get_model_size_in_bytes,
    )
finally:
    _pytree_logger.setLevel(_pytree_log_level)


def select_device() -> torch.device:
    """Use CUDA only when this PyTorch build can execute on the GPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        capability = torch.cuda.get_device_capability()
        required_arch = f"sm_{capability[0]}{capability[1]}"
        built_arches = set(torch.cuda.get_arch_list())
        if built_arches and required_arch not in built_arches:
            raise RuntimeError(
                f"GPU requires {required_arch}, but this PyTorch build "
                f"contains {', '.join(sorted(built_arches))}"
            )
        probe = torch.ones((16, 16), device="cuda")
        _ = probe @ probe
        torch.cuda.synchronize()
    except Exception as error:
        warnings.warn(
            f"CUDA is visible but unusable ({error}). Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")


def make_w8a8_config():
    """Create the dynamic INT8 activation and INT8 weight configuration."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Deprecation: PlainLayout is deprecated.*",
            category=UserWarning,
            module=r"torchao\.dtypes\.utils",
        )
        return Int8DynamicActivationInt8WeightConfig(version=2)


def select_quantization_device(
    preferred_device: torch.device,
) -> torch.device:
    """Check that TorchAO's W8A8 linear path works on the selected device."""
    if preferred_device.type != "cuda":
        return preferred_device

    try:
        probe_model = torch.nn.Linear(
            64, 64, device=preferred_device
        ).eval()
        quantize_(probe_model, make_w8a8_config())
        probe_input = torch.randn(2, 64, device=preferred_device)
        _ = probe_model(probe_input)
        torch.cuda.synchronize()
        del probe_model, probe_input
        torch.cuda.empty_cache()
        return preferred_device
    except Exception as error:
        warnings.warn(
            f"CUDA works for PyTorch but not for this W8A8 path ({error}). "
            "Running the quantization sections on CPU."
        )
        torch.cuda.empty_cache()
        return torch.device("cpu")


def _transformer_macs(
    tokens: int,
    width: int,
    mlp_width: int,
    layers: int,
) -> int:
    projection_macs = 4 * tokens * width**2
    attention_macs = 2 * tokens**2 * width
    mlp_macs = 2 * tokens * width * mlp_width
    return layers * (projection_macs + attention_macs + mlp_macs)


def nominal_encoder_gmacs(model) -> tuple[float, float]:
    """Estimate image- and text-tower MACs from the model architecture."""
    visual = model.visual
    visual_block = visual.transformer.resblocks[0]
    grid_height, grid_width = visual.grid_size
    patch_height, patch_width = visual.patch_size
    visual_width = visual_block.attn.embed_dim
    visual_tokens = grid_height * grid_width + 1
    visual_mlp_width = visual_block.mlp.c_fc.out_features
    visual_layers = len(visual.transformer.resblocks)
    patch_macs = (
        grid_height
        * grid_width
        * visual_width
        * 3
        * patch_height
        * patch_width
    )
    visual_projection_macs = visual_width * visual.output_dim
    image_macs = (
        patch_macs
        + _transformer_macs(
            visual_tokens,
            visual_width,
            visual_mlp_width,
            visual_layers,
        )
        + visual_projection_macs
    )

    text_block = model.transformer.resblocks[0]
    text_width = model.token_embedding.embedding_dim
    text_tokens = model.context_length
    text_mlp_width = text_block.mlp.c_fc.out_features
    text_layers = len(model.transformer.resblocks)
    text_projection_macs = int(np.prod(model.text_projection.shape))
    label_macs = (
        _transformer_macs(
            text_tokens,
            text_width,
            text_mlp_width,
            text_layers,
        )
        + text_projection_macs
    )
    return image_macs / 1e9, label_macs / 1e9


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def encode_preprocessed_images(
    model,
    preprocessed_images: list[torch.Tensor],
) -> np.ndarray:
    with torch.inference_mode():
        return (
            torch.cat([
                model.encode_image(image.unsqueeze(0), normalize=True)
                for image in preprocessed_images
            ])
            .float()
            .cpu()
            .numpy()
        )


def benchmark_image_encoder(
    model,
    preprocessed_images: list[torch.Tensor],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmups):
            for image_tensor in preprocessed_images:
                model.encode_image(image_tensor.unsqueeze(0), normalize=True)

        repeat_seconds = []
        for _ in range(repeats):
            synchronize_device(device)
            start = time.perf_counter()
            for image_tensor in preprocessed_images:
                model.encode_image(image_tensor.unsqueeze(0), normalize=True)
            synchronize_device(device)
            repeat_seconds.append(time.perf_counter() - start)
    return float(np.median(repeat_seconds) / len(preprocessed_images))


def encode_image_collection(
    model,
    preprocess,
    values,
    *,
    batch_size: int,
    device: torch.device,
    decode_image,
) -> np.ndarray:
    batches = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_values = values[start : start + batch_size]
            batch = torch.stack([
                preprocess(decode_image(value)) for value in batch_values
            ]).to(device)
            features = model.encode_image(batch, normalize=True)
            batches.append(features.float().cpu())
            print(
                f"Created embeddings for "
                f"{min(start + batch_size, len(values)):,}/"
                f"{len(values):,} images",
                end="\r",
            )
    print()
    return torch.cat(batches).numpy()


def quantized_weight_parameter_count(model) -> int:
    return sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
        and type(module.weight).__name__ == "Int8Tensor"
    )


def w8a8_producer_manifest(
    *,
    bundle_id: str,
    model_name: str,
    repo_id: str,
    revision: str,
    dataset: dict,
    device_type: str,
) -> dict:
    """Describe the notebook's TorchAO eager visual-tower experiment."""
    return producer_manifest(
        bundle_id=bundle_id,
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        precision="W8A8 dynamic PTQ",
        evidence_type="numerical experiment",
        framework="TorchAO",
        framework_version=TORCHAO_VERSION,
        preprocessing=(
            "OpenCLIP evaluation transform from pinned model config"
        ),
        quantization={
            "scheme": "dynamic W8A8",
            "weights": "per-channel INT8 linear weights",
            "activations": "dynamic INT8 linear activations",
            "scope": "visual tower",
            "calibration": None,
        },
        dataset=dataset,
        backend="Torch eager",
        export=None,
        delegation=None,
        runtime={
            "device_type": device_type,
            "target_runtime_latency_comparable": False,
        },
    )


def load_or_create_embedding_bundle(
    cache_path,
    expected_ids: np.ndarray,
    *,
    manifest: dict,
    create_features,
    recompute: bool,
) -> EmbeddingBundle:
    """Load a matching bundle or create and persist its embeddings."""
    if cache_path.exists() and not recompute:
        try:
            bundle = EmbeddingBundle.load(cache_path, expected_ids)
            print(f"Loaded {bundle.features.shape} from {cache_path}")
            return bundle
        except (KeyError, ValueError, TypeError):
            print(f"Ignoring legacy or stale cache at {cache_path}")

    features = create_features()
    bundle = EmbeddingBundle.create(expected_ids, features, manifest)
    bundle.save(cache_path)
    print(f"Saved {bundle.features.shape} to {cache_path}")
    return bundle


def release_device_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


TORCHAO_VERSION = getattr(torchao, "__version__", "unknown")
