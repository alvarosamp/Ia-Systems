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
    """ Run a subprocess, streaming stderr to loguru and raising onf failure
    """
    logger.info("Running command: {}", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error("Command failed with return code {}: {}", result.returncode, result.stdout)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)