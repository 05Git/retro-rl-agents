import subprocess
from dataclasses import dataclass
from datetime import datetime
from inspect import getmembers
from pathlib import Path
from typing import get_args
from warnings import warn

from retro_rl_agents.domain_models.agent_data import AgentData
from retro_rl_agents.domain_models.env_data import EnvData
from retro_rl_agents.domain_models.service_data import ServiceData


@dataclass
class ConfigData:
    """
    Data model for YAML configs.

    Attributes:
        config_path (Path): Path to actual yaml config.
        working_dir (Path): Path to directory where script is running from.
        save_dir (str): Optional directory name for saving RL models to.
        run_id (str): Optional ID for a specific run.
    """

    config_path: Path

    agent_data: AgentData
    env_data: EnvData
    service_data: ServiceData

    working_dir: Path = Path.cwd().resolve()
    save_dir: str = "trained_agents"
    run_id: str = datetime.now().isoformat(timespec="seconds")

    deterministic: bool = True

    database: Path | None = None

    def __post_init__(self) -> None:
        """
        Format data after __init__
        - Change paths from Strings to Paths
        """
        cls_members_annotated: dict[str, type] = next(
            (m for k, m in getmembers(self) if "__annotations__" in k),
            {}
        )
        # Unpack union types to see what the expected args could be
        unpacked_annotations: dict[str, tuple[type, ...]] = {}
        for k, v in cls_members_annotated.items():
            ann_args: tuple[type, ...] = get_args(v)
            if ann_args:
                unpacked_annotations[k] = ann_args
            else:
                unpacked_annotations[k] = (v,)    
            
        path_fields = (k for k, v in unpacked_annotations.items()
                       if Path in v)
        for f in path_fields:
            if isinstance((f_value := getattr(self, f, None)), str):
                setattr(self, f, Path(f_value))

    @property
    def save_path(self) -> Path:
        return (
            self.working_dir 
            / self.save_dir
            / self.agent_data.model_type
            / self.run_id
        )

    @classmethod
    def generate_timestamp(cls, timespec: str = "seconds") -> str:
        return datetime.now().isoformat(timespec=timespec)

    @classmethod
    def get_sys_info(cls) -> str:
        """
        Return the CPU and GPU info of the system.
        """
        try:
            res = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                 capture_output=True,
                 text=True
            )
        except FileNotFoundError:
            warn_msg = "'nvidia-smi' not found, cannot print sys info."
            warn(warn_msg)
            return warn_msg
        
        retcode = res.returncode
        if retcode == 0:
            sys_info: list[dict[str, str]] = []
            for line in res.stdout.strip().split("\n"):
                d = {}
                keys = ("GPU", "VRAM", "Driver")
                line = line.strip().split(",")[:len(keys)]
                for key, item in zip(keys, line):
                    d[key] = item
                sys_info.append(d)
            return str(sys_info)
        else:
            warn_msg = (
                "Cannot print sys info:"
                f" 'nvidia-smi' return code: {retcode}."
            )
            warn(warn_msg)
            return warn_msg