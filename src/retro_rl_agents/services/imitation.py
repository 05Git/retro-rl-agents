"""
Imitation service - Train agents via imitation learning
"""

import json
import logging
import sqlite3
from typing import Any

import numpy as np
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import CnnRewardNet
from imitation.util import logger as imit_logger

from retro_rl_agents.domain_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(config: ConfigData) -> None:
    imitation_settings: dict[str, Any] = config.service_data.settings
    agent = config.agent_data.agent
    env = config.env_data.env

    logger.info("Running imitation session...")

    imitation_type: str = imitation_settings.pop("type", "bc")
    imitation_log = imit_logger.configure(
        agent.tensorboard_log, ["stdout", "tensorboard"]
    )

    match imitation_type:
        case "bc":
            imitation_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=np.random.default_rng(seed=agent.seed),
                policy=agent.policy,  # type: ignore
                device=agent.device,
                custom_logger=imitation_log,
            )
            model_name = str(imitation_settings["n_epochs"])
        case "gail":
            reward_net = CnnRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                hwc_format=False,
                use_action=False,
                use_next_state=True,
            )
            imitation_trainer = GAIL(
                demonstrations=transitions,
                demo_batch_size=512,
                gen_replay_buffer_capacity=256,
                n_disc_updates_per_round=8,
                venv=env,  # type: ignore
                gen_algo=agent,
                reward_net=reward_net,
                custom_logger=imitation_log,
                allow_variable_horizon=True,
            )
            model_name = str(imitation_settings["n_steps"])
        case _:
            raise ValueError(f"Unexpected 'imitation_type': {imitation_type!r}")

    start_time = config.generate_timestamp()
    try:
        imitation_trainer.train(**imitation_settings)
    except KeyboardInterrupt:
        logger.info(
            "Keyboard interrupt detected, ending imitation training early."
        )
    end_time = config.generate_timestamp()

    config.save_path.mkdir(parents=True, exist_ok=True)

    if config.agent_data.model_path is not None:
        old_name = config.agent_data.model_path.stem
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

    if config.database is None:
        return

    with sqlite3.connect(config.database.resolve()) as conn:
        logger.info("Connecting to %s.", config.database.resolve())
        cur = conn.cursor()
        query = """
            INSERT INTO imitation_runs (
                model_type,
                model_settings,
                model_policy,
                model_path,
                save_path,
                env,
                env_settings,
                tb_path,
                imitation_type,
                avg_loss_final,
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
        # TODO: Refactor after testing + debugging to return relevant rollout stats
        avg_return_final, avg_ep_len_final = config.get_tb_log_final_step_res()
        q_prams = (
            config.agent_data.model_type,
            json.dumps(config.agent_data.serializable_model_settings),
            repr(agent.policy),
            str(config.agent_data.model_path),
            str(config.save_path),
            config.env_data.env_name,
            json.dumps(config.env_data.serializable_env_settings),
            imitation_settings.get("tensorboard_log", ""),
            imitation_type,
            avg_return_final,
            avg_ep_len_final,
            start_time,
            end_time,
            config.config_path.read_text(),
            config.get_sys_info(),
        )
        cur.execute(query, q_prams)
        conn.commit()
