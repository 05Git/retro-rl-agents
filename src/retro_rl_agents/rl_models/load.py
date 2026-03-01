import logging

from importlib import import_module
from typing import Any
from pathlib import Path
from stable_retro import RetroEnv
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)

def load_model(
    model_type: str,
    env: RetroEnv,
    settings_config: dict[str, Any],
    model_path: Path | None
) -> BaseAlgorithm:
    """
    Calls the specified module's 'load_model' method.

    Args:
        model_type (str): Model module (PPO, FuseNet, etc).
        env (RetroEnv): RL env to train/eval/etc on.
        settings_config (dict[str, Any]): Model parameters.
        model_path (Path | None): Optional path to pre-trained model.

    Raises:
        ModuleNotFoundError: Invalid module name.
        AttributeError: Specified module has no 'load_model' method.

    Returns:
        BaseAlgorithm: RL model created by specified module.
    """
    try:
        mod = import_module(f"retro_rl_agents.rl_models.{model_type}")
    except ModuleNotFoundError as e:
        logger.error(e)
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

    try:
        return mod.load_model(
            env=env,
            settings_config=settings_config,
            model_path=model_path
        )
    except TypeError:
        logger.error(
            "Model params contained invalid field(s) and/or value(s): %s",
            str(list(settings_config.items()))
        )
        raise