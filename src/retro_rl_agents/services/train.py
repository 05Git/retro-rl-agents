import json
import logging
import sqlite3
from typing import Any

from retro_rl_agents.domain_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(config: ConfigData) -> None:
    """
    Call an agent's learn method and save the fully trained weights
    to the path specified by the config data.
    """
    train_settings: dict[str, Any] = config.service_data.settings
    agent = config.agent_data.agent
    logger.info("Training...")

    start_time = config.generate_timestamp()
    try:
        train_methods = (
            "learn",  # Stable-Baselines3 train method
        )
        for method in train_methods:
            if hasattr(agent, method):
                getattr(agent, method)(**train_settings)
                break
        else:
            raise AttributeError(
                "No expected 'train' method found in agent."
                " Please make sure the agent implements one of these methods:"
                f" {train_methods}. Agent type: {type(agent).__name__}"
            )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, exiting training early.")
    end_time = config.generate_timestamp()

    config.save_path.mkdir(parents=True, exist_ok=True)

    model_path = config.agent_data.model_path
    if model_path is not None:
        old_name = model_path.stem
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

    with sqlite3.connect(config.database.resolve()) as conn:
        logger.info("Connecting to %s.", config.database.resolve())
        cur = conn.cursor()
        query = """
            INSERT INTO training_runs (
                model_type,
                model_settings,
                model_policy,
                model_path,
                save_path,
                env,
                env_settings,
                tb_path,
                total_timesteps,
                avg_return_final,
                avg_ep_len_final,
                started_at,
                finished_at,
                config_settings,
                sys_settings
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """
        avg_return_final, avg_ep_len_final = config.get_tb_log_final_step_res()
        q_prams = (
            config.agent_data.model_type,
            json.dumps(config.agent_data.serializable_model_settings),
            repr(agent.policy),
            str(config.agent_data.model_path),
            str(config.save_path),
            config.env_data.env_name,
            json.dumps(config.env_data.serializable_env_settings),
            train_settings.get("tensorboard_log", ""),
            train_settings["total_timesteps"],
            avg_return_final,
            avg_ep_len_final,
            start_time,
            end_time,
            config.config_path.read_text(),
            config.get_sys_info(),
        )
        cur.execute(query, q_prams)
        conn.commit()
