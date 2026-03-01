from dataclasses import dataclass, field, fields
from pathlib import Path
from datetime import datetime
from typing import Any

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
    """

    config_path: Path
    model_type: str
    model_path: Path | None = None
    
    working_dir: Path = Path.cwd().resolve()
    save_dir: str = "trained_agents"
    run_id: str = datetime.now().isoformat(timespec="seconds")

    model_settings: dict[str, Any] = field(default_factory=dict)
    service_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Format data after __init__
        - Change paths from Strings to Paths 
        """
        path_fields = (
            "config_path",
            "model_path",
            "working_dir"
        )
        for f in fields(self):
            field_value = getattr(self, f.name)
            if (
                f.name in path_fields
                and isinstance(field_value, str)
            ):
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
            raise KeyError(f"Service '{service_name}' settings not found.")
        return settings
    
    @classmethod
    def generate_timestamp(cls, timespec: str = "seconds") -> str:
        return datetime.now().isoformat(timespec=timespec)