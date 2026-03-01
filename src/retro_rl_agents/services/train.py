import logging

from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Any

from retro_rl_agents.data_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)

def service(agent: BaseAlgorithm, config: ConfigData) -> None:
    """
    Call an agent's learn method and save the fully trained weights
    to the path specified by the config data.

    Args:
        agent (BaseAlgorithm): RL agent to train.
        config (ConfigData): Config containing training params.
    """
    train_settings: dict[str, Any] = config.get_service_settings(NAME)
    logger.info("Training...")
    try:
        agent.learn(**train_settings)
    except KeyboardInterrupt:
        logger.info("Exiting training early.")

    config.save_path.mkdir(parents=True, exist_ok=True)
    save_name = str(train_settings["total_timesteps"])

    save_path = config.save_path / save_name
    try:
        agent.save(path=save_path)
    except Exception as e:
        logger.error(e)
        raise

    logger.info(f"Model data saved to {save_path}.zip")