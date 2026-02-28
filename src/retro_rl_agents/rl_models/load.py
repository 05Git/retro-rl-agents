from importlib import import_module
from typing import Any
from pathlib import Path
from stable_retro import RetroEnv
from stable_baselines3.common.base_class import BaseAlgorithm

def load_model(
    model_type: str,
    env: RetroEnv,
    settings_config: dict[str, Any],
    model_path: Path | None
) -> BaseAlgorithm:
    try:
        mod = import_module(f"retro_rl_agents.rl_models.{model_type}")
    except ModuleNotFoundError as e:
        #TODO: logging
        raise e
    
    if not hasattr(mod, "load_model"):
        err_msg_args = [
            f"Module {mod.__name__} does not contain a 'load_model' method.",
            "Every RL model module must implement a 'load_model' method which",
            "takes a RetroEnv and (optionally) a settings config and",
            "path to a pre-trained model as args. Please ensure this",
            "method is implemented before retrying loading this type of model."
        ]
        raise AttributeError(" ".join(err_msg_args))

    return mod.load_model(
        env=env,
        settings_config=settings_config,
        model_path=model_path
    )