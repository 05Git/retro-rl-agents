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
    env = agent.env
    if env is None:
        raise ValueError(
            "Agent must have env assigned before calling"
            " this service."
        )
    
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs

    while True:
        action, _ = agent.predict(obs, deterministic=config.deterministic)
        result = env.step(action)
        
        if len(result) == 4:
            obs, rew, term, info = result
            done = term
            
        else:
            obs, rew, term, trunc, info = result
            done = term or trunc

        env.render()
        if done:
            break
