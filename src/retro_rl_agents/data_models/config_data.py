from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import CallbackList
from stable_retro import RetroEnv

from retro_rl_agents.callbacks.callback_factory import CallbackFactory
from retro_rl_agents.callbacks.external_cbs import register_external_callbacks


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

    env: RetroEnv

    model_type: str
    model_path: Path | None = None

    working_dir: Path = Path.cwd().resolve()
    save_dir: str = "trained_agents"
    run_id: str = datetime.now().isoformat(timespec="seconds")

    model_settings: dict[str, Any] = field(default_factory=dict)
    service_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    cb_factory: CallbackFactory = CallbackFactory()

    deterministic: bool = True

    def __post_init__(self):
        """
        Format data after __init__
        - Change paths from Strings to Paths
        """
        path_fields = ("config_path", "model_path", "working_dir")
        for f in fields(self):
            field_value = getattr(self, f.name)
            if f.name in path_fields and isinstance(field_value, str):
                setattr(self, f.name, Path(field_value))

    @property
    def save_path(self):
        return self.working_dir / self.save_dir / self.model_type / self.run_id

    @property
    def seed(self) -> int:
        return self.model_settings.get("seed", 0)

    def get_service_settings(self, service_name: str) -> dict[str, Any]:
        settings = self.service_settings.get(service_name)
        if settings is None:
            raise KeyError(f"Service {service_name!r} settings not found.")
        return settings

    @classmethod
    def generate_timestamp(cls, timespec: str = "seconds") -> str:
        return datetime.now().isoformat(timespec=timespec)

    def set_callback(self) -> None:
        for service_name in self.service_settings.keys():
            cb_list: list[dict[str, Any]] | None = (
                self.service_settings[service_name].get("callback")
            )

            if cb_list is None:
                continue

            cb_names = [cfg["type"] for cfg in cb_list]
            register_external_callbacks(
                cb_factory=self.cb_factory,
                callback_list=cb_names
            )

            if "sb3_checkpoint" in cb_names:
                check_idx: int = cb_names.index("sb3_checkpoint")
                cb_list[check_idx]["save_path"] = self.save_path / "checkpoints"

            self.service_settings[service_name]["callback"] = CallbackList([
                self.cb_factory.get_callback(cfg)
                for cfg in cb_list
            ])