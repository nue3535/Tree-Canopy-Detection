"""SSL helpers for HTTPS downloads (fixes macOS Python.framework cert verify failures)."""

from __future__ import annotations

import os
import ssl


def download_insecure_enabled() -> bool:
    for key in ("SAM2_DOWNLOAD_INSECURE", "MASK_RCNN_DOWNLOAD_INSECURE", "PYTORCH_DOWNLOAD_INSECURE"):
        flag = (os.environ.get(key) or "").strip().lower()
        if flag in ("1", "true", "yes"):
            return True
    return False


def ssl_context_for_download() -> ssl.SSLContext:
    if download_insecure_enabled():
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def patch_default_https_context() -> None:
    """Point urllib/torch.hub at certifi's CA bundle (idempotent)."""
    if download_insecure_enabled():
        ssl._create_default_https_context = ssl._create_unverified_context
        return
    try:
        import certifi

        cafile = certifi.where()

        def _factory() -> ssl.SSLContext:
            return ssl.create_default_context(cafile=cafile)

        ssl._create_default_https_context = _factory  # type: ignore[assignment]
    except ImportError:
        pass
