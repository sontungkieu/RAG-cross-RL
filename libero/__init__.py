from __future__ import annotations

import json
import site
from pathlib import Path
from pkgutil import extend_path
from typing import Optional

__path__ = extend_path(__path__, __name__)


def _candidate_uv_checkouts() -> list[Path]:
    candidates: list[Path] = []
    cache_root = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    if not cache_root.exists():
        return candidates

    for checkout_root in sorted(cache_root.glob("*/*")):
        package_root = checkout_root / "libero"
        if (package_root / "libero" / "__init__.py").exists() and (
            package_root / "lifelong" / "__init__.py"
        ).exists():
            candidates.append(package_root)
    return candidates


def _preferred_commit_prefix() -> str:
    for site_dir in site.getsitepackages():
        direct_url = Path(site_dir) / "libero-0.1.0.dist-info" / "direct_url.json"
        if direct_url.exists():
            try:
                payload = json.loads(direct_url.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            commit_id = payload.get("vcs_info", {}).get("commit_id", "")
            if commit_id:
                return commit_id[:7]
    return ""


def _select_checkout() -> Optional[Path]:
    commit_prefix = _preferred_commit_prefix()
    candidates = _candidate_uv_checkouts()
    if commit_prefix:
        for candidate in candidates:
            if candidate.parent.name.startswith(commit_prefix):
                return candidate
    return candidates[0] if candidates else None


_UPSTREAM_PACKAGE_ROOT = _select_checkout()
if _UPSTREAM_PACKAGE_ROOT is not None:
    path_str = str(_UPSTREAM_PACKAGE_ROOT)
    if path_str not in __path__:
        __path__.append(path_str)


def find_upstream_package_root() -> Optional[str]:
    if _UPSTREAM_PACKAGE_ROOT is None:
        return None
    return str(_UPSTREAM_PACKAGE_ROOT)
