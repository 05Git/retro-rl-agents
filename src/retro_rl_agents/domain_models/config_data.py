from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from inspect import getmembers
from pathlib import Path
from typing import Any, get_args

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import FloatSchedule, LinearSchedule

from retro_rl_agents.callbacks.callback_factory import CallbackFactory
from retro_rl_agents.callbacks.external_cbs import register_external_callbacks
from retro_rl_agents.domain_models.agent_data import AgentData
from retro_rl_agents.domain_models.env_data import EnvData
from retro_rl_agents.domain_models.service_data import ServiceData


@dataclass
class ConfigData:
    """
    Data model for YAML configs.

    Attributes:
        config_path (Path): Path to actual yaml config.
        model_type (str): The type of RL model used.
        model_path (Path | None): Optional path to pre-trained model.
        working_dir (Path): Path to directory where script is running from.
        save_dir (str): Optional directory name for saving RL models to.
        run_id (str): Optional ID for a specific run.
        model_settings (dict[str, Any]): RL model parameters.
        train_settings (dict[str, Any]): Training parameters.

    NOTE: This is starting to expand beyond its original scope. Consider
    narrowing down the data and creating a new RunManager class once init
    dev work finishes.
    """

    config_path: Path

    agent_data: AgentData
    env_data: EnvData
    service_data: ServiceData

    working_dir: Path = Path.cwd().resolve()
    save_dir: str = "trained_agents"
    run_id: str = datetime.now().isoformat(timespec="seconds")

    service_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    cb_factory: CallbackFactory = CallbackFactory()
    deterministic: bool = True

    n_envs: int = 1

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
