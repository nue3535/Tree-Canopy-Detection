"""Bootstrap SAM 2 assets into checkpoints_sam2 for API training.

YAML: if ``checkpoints_sam2/<config>.yaml`` is missing, copy from the resolved local source (project or
``sam2`` package). If it already exists, skip (no overwrite).

Weights: if the matching Meta ``.pt`` is missing or too small (likely incomplete), download from Meta CDN;
otherwise skip.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Official Meta release URLs (see facebookresearch/sam2 checkpoints/download_ckpts.sh).
SAM2_V1_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
SAM2_V1_1_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# YAML basename -> (base_url, checkpoint filename on CDN)
SAM2_YAML_TO_CKPT: dict[str, tuple[str, str]] = {
    "sam2_hiera_t.yaml": (SAM2_V1_BASE, "sam2_hiera_tiny.pt"),
    "sam2_hiera_s.yaml": (SAM2_V1_BASE, "sam2_hiera_small.pt"),
    "sam2_hiera_b+.yaml": (SAM2_V1_BASE, "sam2_hiera_base_plus.pt"),
    "sam2_hiera_l.yaml": (SAM2_V1_BASE, "sam2_hiera_large.pt"),
    "sam2.1_hiera_t.yaml": (SAM2_V1_1_BASE, "sam2.1_hiera_tiny.pt"),
    "sam2.1_hiera_s.yaml": (SAM2_V1_1_BASE, "sam2.1_hiera_small.pt"),
    "sam2.1_hiera_b+.yaml": (SAM2_V1_1_BASE, "sam2.1_hiera_base_plus.pt"),
    "sam2.1_hiera_l.yaml": (SAM2_V1_1_BASE, "sam2.1_hiera_large.pt"),
}

USER_AGENT = "Tree-Canopy-Detection/1.0 (SAM2 training bootstrap)"

# Re-download .pt if smaller than this (bytes); complete SAM2 small weights are tens of MB+.
MIN_PT_FILE_BYTES = 1024 * 1024


def _sam2_pkg_root() -> Path | None:
    spec = importlib.util.find_spec("sam2")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def sam2_yaml_search_candidates(project_root: Path) -> list[Path]:
    """Same discovery order as TrainingManager / sam2_workflow (project first, then installed sam2)."""
    ck = project_root / "checkpoints_sam2"
    roots: list[Path] = [
        ck / "sam2_hiera_l.yaml",
        project_root / "sam2_hiera_l.yaml",
        project_root / "configs" / "sam2_hiera_l.yaml",
        project_root / "backend" / "configs" / "sam2_hiera_l.yaml",
        ck / "sam2_hiera_s.yaml",
        project_root / "sam2_hiera_s.yaml",
        project_root / "configs" / "sam2_hiera_s.yaml",
        project_root / "backend" / "configs" / "sam2_hiera_s.yaml",
        ck / "sam2_hiera_b+.yaml",
        ck / "sam2_hiera_t.yaml",
    ]
    pkg = _sam2_pkg_root()
    if pkg:
        roots.extend(
            [
                pkg / "sam2_hiera_l.yaml",
                pkg / "sam2_hiera_s.yaml",
                pkg / "configs" / "sam2" / "sam2_hiera_l.yaml",
                pkg / "configs" / "sam2" / "sam2_hiera_s.yaml",
                pkg / "configs" / "sam2" / "sam2.1_hiera_l.yaml",
                pkg / "configs" / "sam2" / "sam2.1_hiera_s.yaml",
                pkg / "configs" / "sam2" / "sam2.1_hiera_b+.yaml",
                pkg / "configs" / "sam2" / "sam2.1_hiera_t.yaml",
                pkg / "configs" / "sam2" / "sam2_hiera_b+.yaml",
                pkg / "configs" / "sam2" / "sam2_hiera_t.yaml",
            ]
        )
    return roots


def resolve_sam2_yaml(project_root: Path, env: dict[str, str]) -> Path:
    cfg = (env.get("SAM2_MODEL_CFG") or env.get("MODEL_CFG") or "").strip()
    if cfg:
        p = Path(cfg).expanduser()
        if p.is_file():
            return p.resolve()
    for c in sam2_yaml_search_candidates(project_root):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "No SAM2 Hiera YAML found. Install the `sam2` package or add sam2_hiera_*.yaml under checkpoints_sam2."
    )


def _ssl_context_for_download() -> ssl.SSLContext:
    """Prefer certifi's CA bundle (fixes many macOS/Python SSL verify failures)."""
    flag = (os.environ.get("SAM2_DOWNLOAD_INSECURE") or "").strip().lower()
    if flag in ("1", "true", "yes"):
        logger.warning(
            "SAM2_DOWNLOAD_INSECURE is set: SSL verification is disabled for SAM2 checkpoint download."
        )
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def meta_checkpoint_spec_for_yaml(yaml_path: Path) -> tuple[str, str]:
    name = yaml_path.name
    if name not in SAM2_YAML_TO_CKPT:
        raise ValueError(
            f"Unsupported SAM2 config '{name}'. Supported: {', '.join(sorted(SAM2_YAML_TO_CKPT))}. "
            "Set SAM2_CHECKPOINT manually if you use a custom config."
        )
    base, fn = SAM2_YAML_TO_CKPT[name]
    return f"{base}/{fn}", fn


def download_url_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        ctx = _ssl_context_for_download()
        with urllib.request.urlopen(req, timeout=600, context=ctx) as resp, open(partial, "wb") as out:
            chunk = 8 * 1024 * 1024
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
        partial.replace(dest)
    except Exception:
        if partial.exists():
            partial.unlink(missing_ok=True)
        raise


def prepare_sam2_checkpoints_dir(project_root: Path, env: dict[str, str]) -> tuple[Path, Path]:
    """
    Ensure ``checkpoints_sam2`` has the config YAML and matching Meta ``.pt``.

    - YAML: copy from the first resolved local path only when ``checkpoints_sam2/<name>`` does not exist yet;
      if it exists, skip. If the resolved YAML already lives at that path, skip.
    - Weights: download from Meta only when the ``.pt`` is missing or smaller than ``MIN_PT_FILE_BYTES``;
      otherwise skip.

    Returns absolute paths ``(yaml_path, pt_path)`` under ``checkpoints_sam2``.
    """
    yaml_src = resolve_sam2_yaml(project_root, env)
    ckpt_dir = project_root / "checkpoints_sam2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    yaml_dest = (ckpt_dir / yaml_src.name).resolve()
    if yaml_src.resolve() != yaml_dest.resolve():
        if not yaml_dest.is_file():
            shutil.copy2(yaml_src, yaml_dest)

    url, pt_name = meta_checkpoint_spec_for_yaml(yaml_dest)
    pt_dest = (ckpt_dir / pt_name).resolve()
    need_pt = not pt_dest.is_file() or pt_dest.stat().st_size < MIN_PT_FILE_BYTES
    if need_pt:
        try:
            download_url_to_file(url, pt_dest)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP error downloading SAM2 checkpoint ({e.code}): {url}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error downloading SAM2 checkpoint from {url}: {e}") from e

    return yaml_dest, pt_dest
