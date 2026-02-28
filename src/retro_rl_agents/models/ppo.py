from stable_retro import RetroEnv
from stable_baselines3 import PPO
from typing import Any
from pathlib import Path
from retro_rl_agents.utils.constants import DEVICE

def load_model(
    env: RetroEnv,
    settings_config: dict[str, Any] = {},
    model_path: Path | None = None
) -> PPO:
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