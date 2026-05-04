import json
import logging
import sqlite3
from typing import Any

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from retro_rl_agents.domain_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(config: ConfigData) -> None:
    """
    Evaluate the performance of a pretrained model.

    Args:
        agent (BaseAlgorithm): RL agent to train.
        config (ConfigData): Config containing evaluation params.
    """
    if config.agent_data.model_path is None:
        raise ValueError(
            "Cannot eval an agent without a pre-trained model, "
            "please set 'model_path' variable before calling this service."
        )

    logger.info("Evaluating...")
    eval_settings = config.service_data.settings
    start_time = config.generate_timestamp()
    try:
        results = evaluate_policy(
            model=config.agent_data.agent,
            env=config.env_data.env,
            **eval_settings
        )

        if eval_settings.get("return_episode_rewards", False):
            per_ep_returns, per_ep_lens = results
            logger.info("Per episode returns: %s", per_ep_returns)
            logger.info("Per episode lengths: %s", per_ep_lens)
            results_: dict[str, Any] = {
                "per_episode_returns": per_ep_returns,
                "average_return": np.mean(per_ep_returns).item(),
                "std_return": np.std(per_ep_returns).item(),
                "per_episode_lengths": per_ep_lens,
                "average_length": np.mean(per_ep_lens).item(),
                "std_length": np.std(per_ep_lens).item(),
            }
        else:
            avg_return, std_return = results
            logger.info("Average return: %s", avg_return)
            logger.info("Std return: %s", std_return)
            results_: dict[str, Any] = {
                "average_return": avg_return,
                "std_return": std_return,
            }

        results_dict: dict[str, Any] = {
            "model_type": config.agent_data.model_type,
            "model_path": str(config.agent_data.model_path),
            "results": results_,
            "model_settings": config.agent_data.serializable_model_settings,
            "eval_settings": eval_settings,
        }

        save_dir = (
            config.agent_data.model_path.parent / config.generate_timestamp()
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "eval_results.json"

        with open(save_file, mode="wt") as f:
            json.dump(results_dict, f, indent=2)

        logger.info("Eval results saved to %s", save_file)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, exiting evaluation early.")
    
    end_time = config.generate_timestamp()

    if (
        config.database is None
        or "results_" not in locals()
        or "results_dict" not in locals()
    ):
        return
    
    with sqlite3.connect(config.database.resolve()) as conn:
        logger.info("Connecting to %s.", config.database.resolve())
        cur = conn.cursor()
        query = """
            INSERT INTO eval_results (
                model_type,
                model_settings,
                model_policy,
                model_path,
                env,
                env_settings,
                avg_return,
                std_return,
                avg_ep_len,
                std_ep_len,
                full_results,
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
        q_prams = (
            config.agent_data.model_type,
            json.dumps(config.agent_data.serializable_model_settings),
            repr(config.agent_data.agent.policy),
            str(config.agent_data.model_path),
            config.env_data.env_name,
            json.dumps(config.env_data.serializable_env_settings),
            results_.get("average_return", None),    # type: ignore
            results_.get("std_return", None),        # type: ignore
            results_.get("average_length", None),    # type: ignore
            results_.get("std_length", None),        # type: ignore
            results_dict,   # type: ignore
            start_time,
            end_time,
            config.config_path.read_text(),
            config.get_sys_info()
        )
        cur.execute(query, q_prams)
        conn.commit()
