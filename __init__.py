"""
SD HeartMuLa - Multilingual AI Music Generation for ComfyUI
Based on HeartMuLa/HeartLib music foundation model

Generate complete songs with lyrics in English, Chinese, Japanese, Korean, and Spanish.
"""

import sys
import os
import warnings
import importlib.util

# Suppress cosmetic warnings from transformers
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*old version of the checkpointing format.*")
warnings.filterwarnings("ignore", message=".*doesn't directly inherit from.*")
warnings.filterwarnings("ignore", message=".*will NOT inherit from.*")
warnings.filterwarnings("ignore", message=".*_set_gradient_checkpointing.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def import_module_from_path(module_name, file_path):
    """Import a module from an explicit file path to avoid naming conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Ensure heartlib is importable
heartlib_path = os.path.join(current_dir, "sd_heartlib")
if heartlib_path not in sys.path:
    sys.path.insert(0, current_dir)

# Import utilities first (needed by nodes)
hm_paths = import_module_from_path(
    "hm_paths",
    os.path.join(current_dir, "sd_utils", "paths.py")
)
hm_audio_utils = import_module_from_path(
    "hm_audio_utils",
    os.path.join(current_dir, "sd_utils", "audio_utils.py")
)
hm_model_manager = import_module_from_path(
    "hm_model_manager",
    os.path.join(current_dir, "sd_utils", "model_manager.py")
)

# Note: heartmula_wrapper.py removed in v2.0.0 - modular nodes use pipeline directly

# Import nodes
hm_model_loader = import_module_from_path(
    "hm_model_loader",
    os.path.join(current_dir, "sd_nodes", "model_loader.py")
)
hm_conditioning = import_module_from_path(
    "hm_conditioning",
    os.path.join(current_dir, "sd_nodes", "conditioning.py")
)
hm_sampler = import_module_from_path(
    "hm_sampler",
    os.path.join(current_dir, "sd_nodes", "sampler.py")
)
hm_decode = import_module_from_path(
    "hm_decode",
    os.path.join(current_dir, "sd_nodes", "decode.py")
)
hm_tags = import_module_from_path(
    "hm_tags",
    os.path.join(current_dir, "sd_nodes", "tags_builder.py")
)
hm_transcribe = import_module_from_path(
    "hm_transcribe",
    os.path.join(current_dir, "sd_nodes", "transcribe.py")
)

# Get node classes
SD_HeartMuLa_ModelLoader = hm_model_loader.SD_HeartMuLa_ModelLoader
SD_HeartMuLa_Conditioning = hm_conditioning.SD_HeartMuLa_Conditioning
SD_HeartMuLa_Sampler = hm_sampler.SD_HeartMuLa_Sampler
SD_HeartMuLa_Decode = hm_decode.SD_HeartMuLa_Decode
SD_HeartMuLa_TagsBuilder = hm_tags.SD_HeartMuLa_TagsBuilder
SD_HeartMuLa_Transcribe = hm_transcribe.SD_HeartMuLa_Transcribe

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SD_HeartMuLa_ModelLoader": SD_HeartMuLa_ModelLoader,
    "SD_HeartMuLa_Conditioning": SD_HeartMuLa_Conditioning,
    "SD_HeartMuLa_Sampler": SD_HeartMuLa_Sampler,
    "SD_HeartMuLa_Decode": SD_HeartMuLa_Decode,
    "SD_HeartMuLa_TagsBuilder": SD_HeartMuLa_TagsBuilder,
    "SD_HeartMuLa_Transcribe": SD_HeartMuLa_Transcribe,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SD_HeartMuLa_ModelLoader": "SD HeartMuLa Model Loader",
    "SD_HeartMuLa_Conditioning": "SD HeartMuLa Conditioning",
    "SD_HeartMuLa_Sampler": "SD HeartMuLa Sampler",
    "SD_HeartMuLa_Decode": "SD HeartMuLa Decode",
    "SD_HeartMuLa_TagsBuilder": "SD HeartMuLa Tags Builder",
    "SD_HeartMuLa_Transcribe": "SD HeartMuLa Transcribe",
}

# Web directory for frontend extensions
WEB_DIRECTORY = "./web"

# Version info
__version__ = "2.0.2"

# ASCII banner
ascii_art = """
⢀⣀ ⡏⡀    ⣇⣸ ⢀⡀ ⢀⣀ ⡀⣀ ⣰⡀ ⡷⢾ ⡀⢀ ⡇  ⢀⣀
⠣⠼ ⠧⠤    ⠇⠸ ⠣⠭ ⠣⠼ ⠏  ⠘⠤ ⠇⠸ ⠣⠼ ⠧⠤ ⠣⠼
"""
print(f"\033[31m{ascii_art}\033[0m")
print(f"SD HeartMuLa v{__version__} - Multilingual AI Music Generation")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
