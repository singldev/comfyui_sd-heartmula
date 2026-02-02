"""
Model loading and caching for SD HeartMuLa.
Handles downloading, caching, and loading of HeartMuLa models.
"""

import gc
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import torch

# Add heartlib to path
_PACKAGE_ROOT = Path(__file__).parent.parent
_FL_UTILS_DIR = Path(__file__).parent

if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# Import paths module explicitly to avoid relative import issues
def _import_paths():
    module_path = _FL_UTILS_DIR / "paths.py"
    spec = importlib.util.spec_from_file_location("heartmula_paths", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_paths = _import_paths()
get_models_directory = _paths.get_models_directory
get_all_model_paths = _paths.get_all_model_paths

# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}

# Model variant configurations
MODEL_VARIANTS = {
    "3B": {
        "hf_repos": {
            "model": "HeartMuLa/HeartMuLa-oss-3B",
            "codec": "HeartMuLa/HeartCodec-oss",
            "config": "HeartMuLa/HeartMuLaGen",
        },
        "description": "3B parameter model - Good balance of quality and speed",
        "vram_fp16": 12,
        "vram_4bit": 6,
        "max_duration_ms": 240000,
        "languages": ["en", "zh", "ja", "ko", "es"],
    },
    "7B": {
        "hf_repos": {
            "model": "HeartMuLa/HeartMuLa-oss-7B",
            "codec": "HeartMuLa/HeartCodec-oss",
            "config": "HeartMuLa/HeartMuLaGen",
        },
        "description": "7B parameter model - Higher quality (coming soon)",
        "vram_fp16": 24,
        "vram_4bit": 12,
        "max_duration_ms": 240000,
        "languages": ["en", "zh", "ja", "ko", "es"],
    },
}


def get_available_models() -> list:
    """Scan models directory for potential HeartMuLa model folders."""
    models = []
    # Check all registered paths
    for path in get_all_model_paths():
        if path.exists():
            try:
                for item in path.iterdir():
                    if item.is_dir():
                        # Simple heuristic: check for config.json inside
                        if (item / "config.json").exists():
                            # Avoid listing the codec itself as a model choice if possible,
                            # but usually they are distinct. 
                            models.append(item.name)
            except Exception:
                pass
    return sorted(list(set(models)))


def get_variant_list() -> list:
    """Get list of available model variants + scanned folders."""
    defaults = list(MODEL_VARIANTS.keys())
    scanned = get_available_models()
    return sorted(list(set(defaults + scanned)))


def get_variant_info(variant: str) -> dict:
    """Get info for a specific variant."""
    return MODEL_VARIANTS.get(variant, MODEL_VARIANTS["3B"])


def check_models_exist(variant: str) -> bool:
    """
    Check if model files exist locally.

    Args:
        variant: Model variant (3B or 7B)

    Returns:
        True if all required files exist
    """
    all_paths = get_all_model_paths()
    
    for models_dir in all_paths:
        # Check for required directories/files
        required = [
            models_dir / f"HeartMuLa-oss-{variant}",
            models_dir / "HeartCodec-oss",
            models_dir / "tokenizer.json",
            models_dir / "gen_config.json",
        ]

        if all(path.exists() for path in required):
            return True

    return False


def download_models_if_needed(
    variant: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """
    Download model files from HuggingFace if not present.

    Args:
        variant: Model variant (3B or 7B)
        progress_callback: Optional callback for progress updates

    Returns:
        Path to models directory
    """
    # First check if we already have the model in any registered path
    all_paths = get_all_model_paths()
    for path in all_paths:
        required = [
            path / f"HeartMuLa-oss-{variant}",
            path / "HeartCodec-oss",
            path / "tokenizer.json",
            path / "gen_config.json",
        ]
        if all(p.exists() for p in required):
            return path

    from huggingface_hub import snapshot_download

    # Default to the primary writable directory
    models_dir = get_models_directory()
    variant_info = get_variant_info(variant)
    hf_repos = variant_info["hf_repos"]

    total_steps = 3
    current_step = 0

    # Download main model
    model_path = models_dir / f"HeartMuLa-oss-{variant}"
    if not model_path.exists():
        print(f"[SD HeartMuLa] Downloading HeartMuLa-oss-{variant}...")
        snapshot_download(
            repo_id=hf_repos["model"],
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps)

    # Download codec
    codec_path = models_dir / "HeartCodec-oss"
    if not codec_path.exists():
        print("[SD HeartMuLa] Downloading HeartCodec-oss...")
        snapshot_download(
            repo_id=hf_repos["codec"],
            local_dir=str(codec_path),
            local_dir_use_symlinks=False,
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps)

    # Download config files (tokenizer.json, gen_config.json)
    tokenizer_path = models_dir / "tokenizer.json"
    gen_config_path = models_dir / "gen_config.json"
    if not tokenizer_path.exists() or not gen_config_path.exists():
        print("[SD HeartMuLa] Downloading config files...")
        snapshot_download(
            repo_id=hf_repos["config"],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["tokenizer.json", "gen_config.json"],
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps)

    return models_dir


def load_model(
    variant: str = "3B",
    precision: str = "auto",
    use_4bit: bool = False,
    force_reload: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Load HeartMuLaGenPipeline with caching.

    Args:
        variant: Model variant (3B or 7B)
        precision: Precision mode (auto, fp32, fp16, bf16)
        use_4bit: Whether to use 4-bit quantization
        force_reload: Force reload even if cached
        progress_callback: Optional callback for progress updates

    Returns:
        Model info dict containing pipeline and metadata
    """
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Determine dtype
    # Note: MPS supports float16 but NOT bfloat16
    if precision == "auto":
        if device.type in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif precision == "fp32":
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Create cache key
    cache_key = f"{variant}_{device}_{dtype}_{use_4bit}"

    # Return cached model if available
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[SD HeartMuLa] Using cached model: {variant}")
        return _MODEL_CACHE[cache_key]

    # Download models if needed (only for known variants)
    if variant in MODEL_VARIANTS:
        print(f"[SD HeartMuLa] Checking model files for {variant}...")
        models_dir = download_models_if_needed(variant, progress_callback)
        max_duration = MODEL_VARIANTS[variant]["max_duration_ms"]
    else:
        # For custom models, find the path
        models_dir = get_models_directory() # Default fallback
        found = False
        for path in get_all_model_paths():
            if (path / variant).exists():
                models_dir = path
                found = True
                break
        
        if not found:
             print(f"[SD HeartMuLa] Warning: Custom model '{variant}' not found in any registered path. Attempting default...")
        
        print(f"[SD HeartMuLa] Loading custom model: {variant}")
        max_duration = 240000 # Default for custom

    # Import HeartMuLa pipeline
    from heartlib import HeartMuLaGenPipeline

    # Setup quantization config if needed
    bnb_config = None
    if use_4bit:
        # 4-bit quantization only works on CUDA (bitsandbytes requirement)
        if device.type != "cuda":
            print(f"[SD HeartMuLa] WARNING: 4-bit quantization requires CUDA, but device is {device.type}")
            print("[SD HeartMuLa] Disabling 4-bit quantization, using full precision instead")
            use_4bit = False
        else:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("[SD HeartMuLa] Using 4-bit quantization")
            except ImportError:
                print("[SD HeartMuLa] WARNING: bitsandbytes not installed, using full precision")
                bnb_config = None

    # Load the pipeline
    print(f"[SD HeartMuLa] Loading HeartMuLa-{variant} pipeline...")
    try:
        # Try with dtype argument (older versions / local lib)
        pipeline = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=str(models_dir),
            device=device,
            dtype=dtype,
            version=variant,
            bnb_config=bnb_config,
        )
    except TypeError as e:
        # Check if failure is due to 'dtype' arg
        if "dtype" in str(e) or "unexpected keyword argument" in str(e):
            print("[SD HeartMuLa] 'dtype' arg failed, retrying with 'torch_dtype' (HeartMuLa_ComfyUI compatibility)...")
            try:
                # Try with torch_dtype (HeartMuLa_ComfyUI lib)
                pipeline = HeartMuLaGenPipeline.from_pretrained(
                    pretrained_path=str(models_dir),
                    device=device,
                    torch_dtype=dtype,
                    version=variant,
                    bnb_config=bnb_config,
                )
            except TypeError as e2:
                 # If that also fails, try without dtype (some other version)
                 print(f"[SD HeartMuLa] 'torch_dtype' also failed ({e2}). Trying without explicit dtype...")
                 pipeline = HeartMuLaGenPipeline.from_pretrained(
                    pretrained_path=str(models_dir),
                    device=device,
                    version=variant,
                    bnb_config=bnb_config,
                )
                 # Set dtype manually if possible
                 if hasattr(pipeline, 'to') and dtype != torch.float32:
                     pipeline = pipeline.to(dtype=dtype)
        else:
            raise

    # Get actual device from loaded model (may differ if quantization forced CPU)
    # Handle lazy loading where model might be None
    if pipeline.model is not None:
        try:
            actual_device = next(pipeline.model.parameters()).device
            if actual_device != device:
                print(f"[SD HeartMuLa] Note: Model loaded on {actual_device} (requested {device})")
                device = actual_device
        except Exception:
            pass # Use requested device if we can't determine actual
    else:
        # Lazy loading active - assume it will be loaded on requested device
        pass

    # Create model info dict
    model_info = {
        "pipeline": pipeline,
        "version": variant,
        "device": device,
        "dtype": dtype,
        "sample_rate": 48000,
        "max_duration_ms": max_duration,
        "use_4bit": use_4bit,
    }

    # Cache the model
    _MODEL_CACHE[cache_key] = model_info
    print(f"[SD HeartMuLa] Model loaded successfully!")

    return model_info


def clear_model_cache():
    """Clear the model cache and free GPU memory."""
    global _MODEL_CACHE

    for key in list(_MODEL_CACHE.keys()):
        model_info = _MODEL_CACHE.pop(key)
        if "pipeline" in model_info:
            del model_info["pipeline"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("[SD HeartMuLa] Model cache cleared")


def get_cache_info() -> dict:
    """Get information about cached models."""
    return {
        "cached_models": list(_MODEL_CACHE.keys()),
        "num_cached": len(_MODEL_CACHE),
    }


def get_available_vram_gb() -> float:
    """
    Get available VRAM in GB.

    Returns:
        Available VRAM in GB, or 0 if CUDA not available.
    """
    if not torch.cuda.is_available():
        return 0.0

    # Get total and allocated memory
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)

    # Available = total - max(allocated, reserved) - some overhead
    used = max(allocated, reserved)
    available = (total - used) / (1024**3)

    # Account for ~1.5GB system/driver overhead
    available = max(0, available - 1.5)

    return available


def get_recommended_memory_mode(variant: str, use_4bit: bool = False) -> str:
    """
    Auto-detect best memory mode based on available VRAM.

    Args:
        variant: Model variant name
        use_4bit: Whether 4-bit quantization is enabled

    Returns:
        Recommended mode: "normal", "low", or "ultra"
    """
    if not torch.cuda.is_available():
        return "ultra"

    available_vram = get_available_vram_gb()
    # Default to 3B specs if unknown
    variant_info = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["3B"])

    # Get VRAM requirements based on quantization
    if use_4bit:
        vram_normal = variant_info.get("vram_4bit", 6)
        vram_low = max(4, vram_normal - 2)
        vram_ultra = max(3, vram_normal - 3)
    else:
        vram_normal = variant_info.get("vram_fp16", 12)
        vram_low = max(8, vram_normal - 4)
        vram_ultra = max(6, vram_normal - 6)

    if available_vram >= vram_normal + 2:  # Buffer for KV cache
        return "normal"
    elif available_vram >= vram_low:
        return "low"
    else:
        print(f"[SD HeartMuLa] Low VRAM detected ({available_vram:.1f}GB). Using ultra mode.")
        return "ultra"
