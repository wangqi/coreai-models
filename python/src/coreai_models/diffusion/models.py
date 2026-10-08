# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Supported diffusion model families for export.

Diffusion uses HF model IDs directly (no internal registry) — these are
the families known to work end-to-end through the export pipeline.
"""

# Each entry: (family name, example HF model ID, pipeline_type)
SUPPORTED_MODELS: list[tuple[str, str, str]] = [
    ("stable-diffusion-1.x", "runwayml/stable-diffusion-v1-5", "sd"),
    ("stable-diffusion-2.x", "sd2-community/stable-diffusion-2-1", "sd"),
    ("stable-diffusion-3.x", "stabilityai/stable-diffusion-3.5-medium", "sd3"),
    ("flux2", "black-forest-labs/FLUX.2-klein-4B", "flux2"),
    ("wan-t2v-1.3b", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan"),
]


def list_models() -> list[str]:
    """Return the names of supported diffusion model families."""
    return [name for name, _, _ in SUPPORTED_MODELS]


def get_pipeline_type(model_id: str) -> str:
    """Determine the pipeline type for a given HF model ID.

    Returns "sd", "sd3", or "flux2". Raises ValueError for unknown models.
    """
    for _, known_id, ptype in SUPPORTED_MODELS:
        if model_id == known_id:
            return ptype

    # Upstream accepts only the exact ids above, which rejects SD 2.1 base and every SD 1.5 / SD 3.x
    # fine-tune although the export path handles them unchanged. Resolve an unknown id (or a local
    # diffusers folder) by the pipeline class its model_index.json names; SDXL and anything else
    # still raise, because the Swift pipelines do not implement them.
    # wangqi modified 2026-10-08
    class_name = _model_index_class_name(model_id)
    ptype = _PIPELINE_CLASS_TYPES.get(class_name or "")
    if ptype is None and class_name and class_name.startswith("Flux2"):
        ptype = "flux2"
    if ptype is not None:
        return ptype

    raise ValueError(
        f"Unknown diffusion model: '{model_id}'"
        + (f" (model_index.json names {class_name})" if class_name else "")
        + f". Supported models: {[mid for _, mid, _ in SUPPORTED_MODELS]}"
        + " or any repo / local folder whose model_index.json names "
        + f"{sorted(_PIPELINE_CLASS_TYPES)} or a Flux2* pipeline"
    )


# Pipeline classes the Swift runtime implements, by diffusers model_index.json `_class_name`.
# wangqi modified 2026-10-08
_PIPELINE_CLASS_TYPES: dict[str, str] = {
    "StableDiffusionPipeline": "sd",
    "StableDiffusion3Pipeline": "sd3",
}


def _read_model_index(model_id: str) -> dict | None:
    """The parsed model_index.json of a local diffusers folder or a Hub repo, or None.

    Module-level so a workshop wrapper can redirect it to a local tree.
    wangqi modified 2026-10-08
    """
    import json
    from pathlib import Path

    local = Path(model_id).expanduser()
    if local.is_dir():
        index = local / "model_index.json"
        return json.loads(index.read_text()) if index.is_file() else None
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(model_id, "model_index.json")).read_text())
    except Exception:  # offline, gated, missing repo or file: fall through to the ValueError
        return None


def _model_index_class_name(model_id: str) -> str | None:
    """`_class_name` from model_index.json, or None. wangqi modified 2026-10-08"""
    index = _read_model_index(model_id)
    name = index.get("_class_name") if isinstance(index, dict) else None
    return name if isinstance(name, str) else None
