"""
Centralized, type-safe, project settings

Single source of truth for paths, sensor names and label codes. Anything *static*
about the 3w dataset


Env-var convention:
''OIL_WELL__<SECTION>__<FIELD>'' (double underscore separates nested fields). Example
OIL_WELL__PATHS__RAW_DATA_DIR=/path/to/raw/data


Hydra configs in ''configs/' mirror these default and can override them at runtime. The valuesdefined here are the (contract( with th rest of the codebase, and should be considered the default values. They can be overridden by env vars or hydra configs, but the rest of the code should only refer to these settings via the Settings class, and not hardcode any values.))
"""

from __future__ import annotations
from pathlib import Path
from typing import Final
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#Project root is the parent of the src directory
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent.resolve()

SENSORS : Final[tuple[str, ...]] = (
    "P-PDG", # Permanent downhole gauge pressure
    "P-TPT", # Temperature/pressure transducer pressure
    "T-TPT", # Temperature/pressure transducer temperature
    "P-MON-CKP", # Monitor choke pressure(monitoring)
    "T-JUS-CKP", # Just choke temperature
    "P-JUS-CKGL", # Downstream gas-lift choke pressure
    "T-JUS-CKGL", # Downstream gas-lift choke temperature
    "QGL", # Gas-lift flow rate
)

#Name of the target label column in the dataframes
TARGET_LABEL: Final[str] = "class"

#Project focues : Spurius Closure of DHSV (Downhole Safety valve)
# 0 : normal steady-state operation -> Indica operação normal, poço sem anomalias
# 2 : DHSV closure event(anomaly, steady-state) -> Indiica evento de fechamento, ou seja, anomalia detectada(estado estavel, porem anomalo)
# 102: DHSV closure transient ( the run-up into the anomaly) -> Indica periodo de transição(transiente) que antecede o fechamento da DHSV, ou seja, o fechament

LABEL_NORMAL : Final[int] = 0
LABEL_DHSV_FAULT : Final[int] = 2
LABEL_DHSV_TRANSIENT : Final[int] = 102

#All labels we keep when filtering raw data for the DHSV problem
TARGET_LABELS : Final[tuple[int, ...]] = (
    LABEL_NORMAL,
    LABEL_DHSV_FAULT,
    LABEL_DHSV_TRANSIENT,
)

EVENT_DIRS_FOR_DHSV : Final[tuple[str, ...]] = ('0', '2')


#Pydantic seting modelos

class PathsSettings(BaseSettings):
    """Filesystem paths. All paths are absolute bt the time you read them"""
    
    model_config = SettingsConfigDict(
        env_prefix = 'OIL_WELL__PATHS__',
        env_file = '.env',
        env_file_encoding = 'utf-8',
        extra = 'ignore',
    )
    data_root : Path = Field(default = PROJECT_ROOT / 'data', description = 'Root directory for all data'   )
   
    @property
    def raw_dir(self) -> Path:
       return self.data_root / 'raw'
   
    @property
    def interim_dir(self) ->Path:
        return self.data_root / 'interim'
    
    @property
    def processed_dir(self) -> Path:
        return self.data_root / 'processed'
    
    def ensure(self) -> None:
        for p in (self.raw_dir, self.interim_dir, self.processed_dir):
            p.mkdir(parents = True, exist_ok = True)

class DatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix = 'OIL_WELL__DATASET__',
        env_file = '.env',
        env_file_encoding = 'utf-8',
        extra = 'ignore',
    )
    repo_url : str = 'https://github.com/petrobras/3w.git'
    git_ref : str = 'main'
    repo_subdir : str = 'raw/3w'
    dataset_subdir : str = 'dataset'
    
    
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix = 'OIL_WELL__',
        env_file = '.env',
        env_file_encoding = 'utf-8',
        extra = 'ignore',
    )
    paths: PathsSettings = Field(default_factory = PathsSettings)
    dataset : DatasetSettings = Field(default_factory = DatasetSettings)
    log_level : str = 'INFO'
    
    @property
    def repo_path(self) -> Path:
        return self.paths.data_root / self.dataset.repo_subdir
    @property
    def dataset_path(self) -> Path:
        return self.repo_path / self.dataset.dataset_subdir
    
#Convenience function to get settings instance
settings = Settings()