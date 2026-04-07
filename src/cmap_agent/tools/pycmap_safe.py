from __future__ import annotations

import os
from typing import Optional


def patch_pycmap_config_path(path: Optional[str] = None) -> str:
    """Patch pycmap to write/read its config from a safe location (default: /tmp).

    pycmap persists auth/baseURL in a CSV config. In containers or restricted environments,
    the default location may be read-only. We patch pycmap.common.config_path to point to
    /tmp (or a caller-provided path).
    """
    from pycmap import common

    tmp_cfg = path or os.environ.get("PYCMAP_CONFIG_PATH", "/tmp/pycmap_config.csv")

    def _config_path():
        return tmp_cfg

    try:
        common.config_path = _config_path  # type: ignore[attr-defined]
    except Exception:
        pass

    os.environ["PYCMAP_CONFIG_PATH"] = tmp_cfg
    return tmp_cfg


def patch_pycmap_halt() -> None:
    """Prevent pycmap from calling sys.exit() (which can kill the web worker).

    pycmap.common.halt() calls sys.exit(). In a FastAPI server, that can abort a worker
    mid-request and look like the request is "hanging". We patch halt() to raise a
    regular exception instead.
    """
    from pycmap import common

    if getattr(common, "_cmap_agent_patched_halt", False):
        return

    def _halt_raise(msg):  # type: ignore[no-redef]
        raise ValueError(str(msg))

    try:
        common.halt = _halt_raise  # type: ignore[attr-defined]
        common._cmap_agent_patched_halt = True  # type: ignore[attr-defined]
    except Exception:
        pass


def make_pycmap_api(token: str, base_url: Optional[str] = None):
    """Create a pycmap.API client with safe config path + halt behavior."""
    import warnings

    import pycmap

    # Silence noisy deprecation warnings emitted inside pycmap's REST helpers.
    # We still want to see our own warnings, so we scope this to that module.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pycmap\\.rest")

    patch_pycmap_config_path()
    patch_pycmap_halt()

    if base_url:
        return pycmap.API(token=token, baseURL=base_url)
    return pycmap.API(token=token)
