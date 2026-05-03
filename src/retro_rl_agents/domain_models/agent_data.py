from copy import deepcopy
from dataclasses import dataclass, field
from inspect import getmembers
from pathlib import Path
from typing import Any, get_args
from warnings import warn

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import (
    FloatSchedule,
    LinearSchedule,
)
from tbparse import SummaryReader


@dataclass
class AgentData:
    """
    Dataclass for holding info about the RL model.
    """

    model_type: str
    agent: BaseAlgorithm | Any # Abstract into protocols later
    model_path: Path | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Format data after __init__
        - Change paths from Strings to Paths
        """
        cls_members_annotated: dict[str, type] = next(
            (m for k, m in getmembers(self) if "__annotations__" in k),
            {}
        )
        # Unpack union types to see what the expected args should be
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

    def get_tb_log_final_step_res(self) -> tuple[float | None, float | None]:
        tb_log_path: str | None = self.model_settings.get("tensorboard_log")
        if tb_log_path is None:
            warn("Tensorboard log path not set. Returning 'None'.")
            return None, None

        tb_reader = SummaryReader(tb_log_path)
        scalars_df = tb_reader.scalars

        final_values = scalars_df.groupby("tag").last().reset_index()
        final_values = final_values[final_values["tag"].str.contains("rollout")]

        try:
            avg_return_final = (
                final_values
                .loc[final_values["tag"] == "rollout/ep_rew_mean", "value"]
                .iloc[0]
            )
            avg_ep_len_final = (
                final_values
                .loc[final_values["tag"] == "rollout/ep_len_mean", "value"]
                .iloc[0]
            )
        except IndexError:
            warn(
                "Unable to extract final returns and ep lens"
                f" from {tb_log_path}. Returning 'None'."
            )
            return None, None
        
        return float(avg_return_final), float(avg_ep_len_final)