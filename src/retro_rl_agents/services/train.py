import logging
from typing import Any

from stable_baselines3.common.base_class import BaseAlgorithm

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
        logger.info("Keyboard interrupt detected, exiting training early.")

    config.save_path.mkdir(parents=True, exist_ok=True)

    if config.model_path is not None:
        old_name = config.model_path.stem
        try:
            save_name = str(train_settings["total_timesteps"] + int(old_name))
        except ValueError:
            save_name = old_name + config.generate_timestamp()
    else:
        save_name = str(train_settings["total_timesteps"])

    save_path = config.save_path / save_name
    try:
        agent.save(path=save_path)
    except Exception as e:
        logger.error(e)
        raise

    logger.info(f"Model data saved to {save_path}.zip")
