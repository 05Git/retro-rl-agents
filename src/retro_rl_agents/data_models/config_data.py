from dataclasses import dataclass, field, fields
from pathlib import Path
from datetime import datetime
from typing import Any

@dataclass
class ConfigData:
    """
    Docstring
    """

    config_path: Path
    model_type: str
    model_path: Path | None = None
    
    working_dir: Path = Path.cwd().resolve()
    save_dir: str = "trained_agents"
    run_id: str = datetime.now().isoformat(timespec="seconds")

    # TODO: Model more thoroughly
    model_settings: dict[str, Any] = field(default_factory=dict)
    train_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Format data after __init__
        - Change paths from Strings to Paths 
        """
        for f in fields(self):
            field_value = getattr(self, f.name)
            if (
                f.name in ("config_path", "model_path", "working_dir")
                and isinstance(field_value, str)
            ):
                setattr(self, f.name, Path(field_value))

    @property
    def save_path(self):
        return self.working_dir / self.save_dir / self.model_type / self.run_id
    