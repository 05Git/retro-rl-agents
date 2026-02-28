from importlib import import_module
from typing import Any
from stable_baselines3.common.base_class import BaseAlgorithm

def call_service(
    service_name: str,
    agent: BaseAlgorithm,
    service_config: dict[str, Any]
) -> None:
    try:
        mod = import_module(f"retro_rl_agents.services.{service_name}")
    except ModuleNotFoundError as e:
        #TODO: logging
        raise e
    
    if not hasattr(mod, "service"):
        # TODO: Abstract err msg args
        err_msg_args = [
            f"Module {mod.__name__} does not contain a 'service' method.",
            "All service modules must contain a 'service' method which takes",
            "an agent and a config as arguments. Please ensure this method is",
            "implemented before callingn this service again."
        ]
        raise AttributeError(" ".join(err_msg_args))
    
    mod.service(agent=agent, config=service_config)