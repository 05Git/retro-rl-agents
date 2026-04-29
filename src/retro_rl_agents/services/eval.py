import json
import logging
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from retro_rl_agents.domain_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(agent: BaseAlgorithm, config: ConfigData) -> None:
    """
    Evaluate the performance of a pretrained model.

    Args:
        agent (BaseAlgorithm): RL agent to train.
        config (ConfigData): Config containing evaluation params.
    """
    if config.model_path is None:
        raise ValueError(
            "Cannot eval an agent without a pre-trained model, "
            "please set 'model_path' variable before calling this service."
        )

    eval_settings: dict[str, Any] = config.get_service_settings(NAME)
    logger.info("Evaluating...")
    try:
        eval_env = agent.env
        if eval_env is None:
            raise ValueError(
                "Agent's env is set to 'None'."
                " Must set agent's env before calling the eval service."
            )
        results = evaluate_policy(model=agent, env=eval_env, **eval_settings)

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
            "model_type": config.model_type,
            "model_path": str(config.model_path),
            "results": results_,
            "model_settings": config.serializable_model_settings,
            "eval_settings": eval_settings,
        }

        save_dir = config.model_path.parent / config.generate_timestamp()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "eval_results.json"

        with open(save_file, mode="wt") as f:
            json.dump(results_dict, f, indent=2)

        logger.info("Eval results saved to %s", save_file)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, exiting evaluation early.")
