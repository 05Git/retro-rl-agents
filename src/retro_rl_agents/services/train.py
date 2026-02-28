from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Any

from retro_rl_agents.data_models.config_data import ConfigData

def service(agent: BaseAlgorithm, config: ConfigData) -> None:    
    train_settings: dict[str, Any] = config.train_settings
    agent.learn(**train_settings)

    config.save_path.mkdir(parents=True, exist_ok=True)
    save_name = str(train_settings["total_timesteps"])

    agent.save(path=config.save_path / save_name)