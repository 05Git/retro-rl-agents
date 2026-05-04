from copy import deepcopy
from dataclasses import dataclass, field
from inspect import getmembers
from pathlib import Path
from typing import Any, get_args

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import (
    FloatSchedule,
    LinearSchedule,
)


@dataclass
class AgentData:
    """
    Dataclass for holding info about the RL model.
    """

    model_type: str
    agent: BaseAlgorithm | Any  # Abstract into protocols later
    model_path: Path | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Format data after __init__
        - Change paths from Strings to Paths
        """
        cls_members_annotated: dict[str, type] = next(
            (m for k, m in getmembers(self) if "__annotations__" in k), {}
        )
        # Unpack union types to see what the expected args should be
        unpacked_annotations: dict[str, tuple[type, ...]] = {}
        for k, v in cls_members_annotated.items():
            ann_args: tuple[type, ...] = get_args(v)
            if ann_args:
                unpacked_annotations[k] = ann_args
            else:
                unpacked_annotations[k] = (v,)

        path_fields = (k for k, v in unpacked_annotations.items() if Path in v)
        for f in path_fields:
            if isinstance((f_value := getattr(self, f, None)), str):
                setattr(self, f, Path(f_value))

    @property
    def serializable_model_settings(self) -> dict[str, Any]:
        serializable_settings = deepcopy(self.model_settings)
        non_serializable = (
            FloatSchedule,
            LinearSchedule,
        )
        non_serializable_items: list[tuple] = [
            i
            for i in serializable_settings.items()
            if isinstance(i[1], non_serializable)
        ]
        for k, v in non_serializable_items:
            serializable_settings[k] = repr(v)

        return serializable_settings
