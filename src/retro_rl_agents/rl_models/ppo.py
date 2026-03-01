from stable_retro import RetroEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import FloatSchedule, LinearSchedule
from typing import Any
from pathlib import Path
from retro_rl_agents.utils.constants import DEVICE

def load_model(
    env: RetroEnv,
    settings_config: dict[str, Any] = {},
    model_path: Path | None = None
) -> PPO:
    schedule_fields = (
        "learning_rate",
        "clip_range",
        "clip_range_vf"
    )
    for field in schedule_fields:
        field_value = settings_config.get(field)

        if isinstance(field_value, list):
            match len(field_value):
                case 3:
                    settings_config[field] = FloatSchedule(
                        LinearSchedule(*field_value)
                    )

                case 2:
                    settings_config[field] = FloatSchedule(
                        LinearSchedule(*field_value, end_fraction=1)
                    )
                    
                case _:
                    raise ValueError(
                        f"Expected {field} list to have 2 or 3 elements, "
                        f"received {len(field_value)} instead. Ensure {field} "
                        "list contains elements for start value, end value, "
                        "and (optional) an end fraction value between 0 and 1."
                    )

        elif isinstance(field_value, dict):
            settings_config[field] = FloatSchedule(
                LinearSchedule(**field_value)
            )
        
        elif (
            isinstance(field_value, (int, float))
            or field_value is None
        ):
            continue
        
        else:
            raise TypeError(
                f"Expected {field} to be a list, dict, int or float, "
                f"received value of type {type(field).__name__}."
            )

    """Load an SB3 PPO model"""
    if model_path is not None:
        return PPO.load(
            path=model_path,
            env=env,
            device=DEVICE,
            **settings_config
        )
    
    return PPO(
        env=env,
        device=DEVICE,
        **settings_config
    )