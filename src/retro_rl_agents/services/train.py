import logging
import sqlite3
from typing import Any

from stable_baselines3.common.base_class import BaseAlgorithm

from retro_rl_agents.domain_models.config_data import ConfigData

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

    start_time = config.generate_timestamp()
    try:
        agent.learn(**train_settings)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, exiting training early.")
    end_time = config.generate_timestamp()

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
        logger.error(f"Unable to save model: {e}")
        raise

    logger.info(f"Model data saved to {save_path}.zip")

    if config.database is None:
        return
    
    with sqlite3.connect(config.database) as conn:
        cur = conn.cursor()
        query = """
            INSERT INTO training_runs (
                model_type,
                model_settings,
                network_layers,
                model_path,
                save_path,
                env,
                env_settings,
                tb_path,
                total_timesteps,
                avg_return_final,
                std_return_final,
                avg_ep_len_final,
                std_ep_len_final,
                started_at,
                finished_at,
                sys_settings
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
        """
        q_prams = (
            config.model_type,
            config.serializable_model_settings,
            agent.policy,
            config.model_path,
            config.save_path,
            ...,
            ..., # Will refactor to use EnvModel rather than agent.env
            train_settings.get("tensorboard_log", ""),
            train_settings["total_timesteps"],
            ...,
            ...,
            ...,
            ..., # Figure out how to extract data from TB
            start_time,
            end_time,
            ...  # Call nvidia-smi through subprocess
        )
        cur.execute(query, q_prams)
        conn.commit()