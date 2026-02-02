"""
Path utilities for SD HeartMuLa.
Handles model directory resolution and package paths.
"""

import os
from pathlib import Path


def get_models_directory() -> Path:
    """
    Get the ComfyUI models/HeartMuLa directory.
    Creates the directory if it doesn't exist.

    Returns:
        Path to models/HeartMuLa directory
    """
    try:
        import folder_paths
        # Register "HeartMuLa" if not present, pointing to standard location
        # Use CamelCase to match HeartMuLa_ComfyUI convention
        if "HeartMuLa" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("HeartMuLa", str(Path(folder_paths.models_dir) / "HeartMuLa"))
        
        # Get paths (returns a list, first one is primary/writable usually)
        paths = folder_paths.get_folder_paths("HeartMuLa")
        if paths:
            base = Path(paths[0])
        else:
            base = Path(folder_paths.models_dir) / "HeartMuLa"
            
    except Exception:
        # Fallback for non-ComfyUI environments
        base = Path(__file__).parent.parent.parent.parent.parent / "models" / "HeartMuLa"

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_all_model_paths() -> list[Path]:
    """
    Get all registered HeartMuLa model paths.
    
    Returns:
        List of Paths where models might be found.
    """
    try:
        import folder_paths
        if "HeartMuLa" not in folder_paths.folder_names_and_paths:
             folder_paths.add_model_folder_path("HeartMuLa", str(Path(folder_paths.models_dir) / "HeartMuLa"))
             
        paths = folder_paths.get_folder_paths("HeartMuLa")
        return [Path(p) for p in paths]
    except Exception:
        return [get_models_directory()]


def get_heartlib_path() -> Path:
    """
    Get path to the bundled heartlib source.

    Returns:
        Path to heartlib directory within the node pack
    """
    return Path(__file__).parent.parent / "sd_heartlib"


def get_package_root() -> Path:
    """
    Get the root directory of the SD-HeartMuLa package.

    Returns:
        Path to ComfyUI_SD-HeartMuLa directory
    """
    return Path(__file__).parent.parent


def ensure_heartlib_in_path():
    """
    Ensure the heartlib directory is in Python path for imports.
    """
    import sys
    heartlib_parent = str(get_package_root())
    if heartlib_parent not in sys.path:
        sys.path.insert(0, heartlib_parent)
