"""
Download the 3w dataset 

The 3w repo ships a python toolkit and the dataset itself under.
We only need the data, so this module uses **git-sparse_chackout** to fetch just thath subdirectory at pinned ref. Re-runs are idempotent ( fetch + reset)

CLI 
---
The hydra entry point lives at the bottom of this file:

    python -m data.download
    python -m data.download data.event_dirs = [0,2,5]
    pyhon -m data.download +force=true
    
Or via the installed shortcurt::
    ow-download
"""

from __future__ import annotations
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
import hydra
from loguru import logger
from omegaconf import DictConfig
from core.settings import EVENTS_DIRS_FOR_DHSV, settings

#Library
def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, streaming stderr to loguru and raising on failure."""
    logger.debug("$ {}", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.error("Command failed: {}", " ".join(cmd))
        if proc.stdout:
            logger.error("stdout:\n{}", proc.stdout)
        if proc.stderr:
            logger.error("stderr:\n{}", proc.stderr)
        raise RuntimeError(f"git command failed with exit {proc.returncode}")

def _ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is not available in PATH. Please install git to use this script.")

def download_3w(
    event_dirs: Iterable[str] = EVENT_DIRS_FOR_DHSV,
    *,
    force: bool = False,
    repo_url: str | None = None,
    git_ref: str | None = None,
    target_dir: Path | None = None,
) -> Path:
    """Sparse-clone the 3W repo and pin to ``git_ref``.
 
    Parameters
    ----------
    event_dirs
        Names of subfolders inside ``dataset/`` to include in the sparse
        checkout. Defaults to the DHSV-relevant folders (``0`` and
        ``2``). Pass ``("*",)`` to fetch every event type.
    force
        If True, wipe ``target_dir`` before cloning. Use this after
        bumping ``git_ref``.
    repo_url, git_ref, target_dir
        Optional overrides. Default to whatever ``settings`` resolves to
        from env / .env.
 
    Returns
    -------
    Path
        Absolute path to the local clone.
    """
    _ensure_git_available()
 
    repo_url = repo_url or settings.dataset.repo_url
    git_ref = git_ref or settings.dataset.git_ref
    target = target_dir or settings.repo_path
    settings.paths.ensure()
 
    if target.exists():
        if force:
            logger.warning("Wiping existing clone at {}", target)
            shutil.rmtree(target)
        else:
            logger.info(
                "Clone already exists at {}. Pulling instead. Pass force=true "
                "to re-clone.",
                target,
            )
            _run(["git", "fetch", "--depth", "1", "origin", git_ref], cwd=target)
            _run(["git", "checkout", git_ref], cwd=target)
            _run(["git", "reset", "--hard", f"origin/{git_ref}"], cwd=target)
            return target
 
    target.parent.mkdir(parents=True, exist_ok=True)
 
    # 1) Clone with no blobs and sparse mode enabled.
    logger.info("Cloning {} into {} (sparse, depth=1, ref={})",
                repo_url, target, git_ref)
    _run([
        "git", "clone",
        "--filter=blob:none",
        "--depth", "1",
        "--branch", git_ref,
        "--sparse",
        repo_url,
        str(target),
    ])
 
    # 2) Configure sparse paths. We always include the dataset root files
    #    (folds, README, etc.) plus each requested event dir.
    sparse_paths = ["dataset/*"] if "*" in event_dirs else [
        "dataset/folds",
        *(f"dataset/{d}" for d in event_dirs),
    ]
    logger.info("Sparse-checkout paths: {}", sparse_paths)
    _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=target)
 
    # 3) Sanity check.
    ds_dir = target / settings.dataset.dataset_subdir
    if not ds_dir.exists():
        raise RuntimeError(f"Expected dataset directory not found: {ds_dir}")
 
    n_parquet = sum(1 for _ in ds_dir.rglob("*.parquet"))
    logger.success("3W ready at {} ({} parquet files)", ds_dir, n_parquet)
    return target
 


#Hydra entry point
@hydra.main(version_base = None, config_path = '../../configs', config_name = 'config')
def main(cfg: DictConfig) -> None:
    """Hydra entry point. See module docstring for usage."""
    logger.info("Config:\n{}", cfg)
    download_3w(
        event_dirs=list(cfg.data.event_dirs),
        force=bool(cfg.get("force", False)),
        repo_url=cfg.data.repo_url,
        git_ref=cfg.data.git_ref,
        target_dir=Path(cfg.paths.repo_dir),
    )
    
if __name__ == "__main__":
    main()