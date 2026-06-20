from __future__ import annotations

from pathlib import Path

def require_file(path: str | Path, purpose: str) -> Path:
    """
    Checks if a file exists. If it does, returns its Path.
    If it is missing, raises a FileNotFoundError explaining the purpose and how to generate it.
    
    Args:
        path: Path to the required file.
        purpose: Explanation of why this file is needed.
        
    Returns:
        Path object if file exists.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required file '{p.name}' is missing (Path: {p.absolute()}).\n"
            f"Purpose: {purpose}\n"
            f"Integrity Policy: Fake/random fallback data is strictly forbidden.\n"
            f"Action: Please generate or provide the real artifact through the documented canonical pipeline."
        )
    return p
