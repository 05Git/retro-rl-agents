import logging

from stable_baselines3.common.base_class import BaseAlgorithm

from retro_rl_agents.data_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(agent: BaseAlgorithm, config: ConfigData) -> None:
    """
    Observe an agent's behaviour by watching it play the dang game.

    Args:
        agent (BaseAlgorithm): RL agent to train.
        config (ConfigData): Config containing service params.
    """
    env = config.env
    obs, _ = env.reset()
    while True:
        action = agent.predict(obs, deterministic=config.deterministic)
        obs, rew, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            break
