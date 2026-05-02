"""
Imitation service - Train agents via imitation learning
"""

import logging
from typing import Any

import numpy as np
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import CnnRewardNet
from imitation.util import logger as imit_logger
from stable_baselines3.common.base_class import BaseAlgorithm

from retro_rl_agents.domain_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(agent: BaseAlgorithm, config: ConfigData) -> None:
    imitation_settings: dict[str, Any] = config.get_service_settings(NAME)
    logger.info("Running imitation session...")

    if agent.env is None:
        raise ValueError(
            "Agent 'env' property is None. Please set the agent's"
            " env before calling the imitation service."
        )

    imitation_type: str = imitation_settings.pop("type", "bc")
    imitation_log = imit_logger.configure(
        agent.tensorboard_log,
        ["stdout", "tensorboard"]
    )

    match imitation_type:
        case "bc":
            imitation_trainer = bc.BC(
                observation_space=agent.env.observation_space,
                action_space=agent.env.action_space,
                demonstrations=transitions,
                rng=np.random.default_rng(seed=agent.seed),
                policy=agent.policy, # type: ignore
                device=agent.device,
                custom_logger=imitation_log,
            )
            model_name = str(imitation_settings["n_epochs"])
        case "gail":
            reward_net = CnnRewardNet(
                observation_space=agent.env.observation_space,
                action_space=agent.env.action_space,
                hwc_format=False,
                use_action=False,
                use_next_state=True,
            )
            imitation_trainer = GAIL(
                demonstrations=transitions,
                demo_batch_size=512,
                gen_replay_buffer_capacity=256,
                n_disc_updates_per_round=8,
                venv=agent.env,
                gen_algo=agent,
                reward_net=reward_net,
                custom_logger=imitation_log,
                allow_variable_horizon=True,
            )
            model_name = str(imitation_settings["n_steps"])
        case _:
            raise ValueError(
                f"Unexpected 'imitation_type': {imitation_type!r}"
            )
        
    try:
        imitation_trainer.train(**imitation_settings)
    except KeyboardInterrupt:
        logger.info(
            "Keyboard interrupt detected, ending imitation training early."
        )

    config.save_path.mkdir(parents=True, exist_ok=True)

    if config.model_path is not None:
        old_name = config.model_path.stem
        try:
            save_name = str(int(model_name) + int(old_name))
        except ValueError:
            save_name = old_name + config.generate_timestamp()
    else:
        save_name = model_name

    save_path = config.save_path / save_name
    try:
        agent.save(path=save_path)
    except Exception as e:
        logger.error(e)
        raise

    logger.info(f"Model data saved to {save_path}.zip")