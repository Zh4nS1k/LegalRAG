"""
Hybrid Connectivity Hook utilities.
Strict 2-second timeout for connectivity checks—no hangs.
"""

import socket
import sys
from pathlib import Path

CONNECTIVITY_TIMEOUT_SEC = 2.0


def is_internet_available(timeout: float = CONNECTIVITY_TIMEOUT_SEC) -> bool:
    """
    Ping 8.8.8.8 or huggingface.co with strict timeout.
    Returns True if reachable, False otherwise. Never blocks > timeout seconds.
    """
    for host, port in [("8.8.8.8", 53), ("huggingface.co", 443)]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.timeout, OSError, socket.error):
            continue
    return False


def is_cache_populated(cache_dir: str) -> bool:
    """
    Check if .models_cache has any cached model data.
    Returns True if directory exists and contains files/subdirs (e.g. models--*).
    """
    path = Path(cache_dir)
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False
