import logging

from retro_rl_agents.domain_models.config_data import ConfigData

NAME = __name__.split(".")[-1]
logger = logging.getLogger(NAME)


def service(config: ConfigData) -> None:
    """
    Observe an agent's behaviour by watching it play the dang game.
    """
    env = config.env_data.env
    agent = config.agent_data.agent
    
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs

    while True:
        action, _ = agent.predict(obs, deterministic=config.deterministic) # type: ignore
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
