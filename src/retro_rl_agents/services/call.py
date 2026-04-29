import logging
from importlib import import_module

from stable_baselines3.common.base_class import BaseAlgorithm

from retro_rl_agents.domain_models.config_data import ConfigData

logger = logging.getLogger(__name__)


def call_service(
    service_name: str, agent: BaseAlgorithm, config: ConfigData
) -> None:
    """
    Calls a service module's 'service' function.
    These are intended to perform specific functions for RL agents,
    e.g. the 'train' service runs an agent's learn method and
    saves it to the specified directory.

    Args:
        service_name (str): Service module to be imported.
        agent (BaseAlgorithm): RL agent loaded by rl_models.load_agent
        config (ConfigData): Config data model.

    Raises:
        ModuleNotFoundError: Invalid service module name.
        AttributeError: Service module has no 'service' method.
    """
    try:
        mod = import_module(f"retro_rl_agents.services.{service_name}")
    except ModuleNotFoundError as e:
        logger.error(e)
        raise e

    if not hasattr(mod, "service"):
        # TODO: Abstract err msg args
        err_msg_args = [
            f"Module {mod.__name__} does not contain a 'service' method.",
            "All service modules must contain a 'service' method which takes",
            "an agent and a config as arguments. Please ensure this method is",
            "implemented before calling this service again.",
        ]
        raise AttributeError(" ".join(err_msg_args))

    mod.service(agent=agent, config=config)
