import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from gymnasium.wrappers import (
    GrayscaleObservation,
    NormalizeReward,
    ResizeObservation,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_retro import make

from retro_rl_agents.cli.arguments import get_args
from retro_rl_agents.data_models.config_data import ConfigData
from retro_rl_agents.rl_models.load import load_model
from retro_rl_agents.services.call import call_service
from retro_rl_agents.utils.constants import (
    DEVICE,
    GAME_NAME_MAP,
    LOG_DIR,
    VALID_SERVICES,
)

logger = logging.getLogger(__name__)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logfile = LOG_DIR / (datetime.now().isoformat(timespec="seconds") + ".log")

logging.basicConfig(
    format="[%(levelname)s|%(asctime)s|%(name)s] %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(filename=logfile),
    ],
)


def main():
    """Entry point for calling services like 'train' or 'eval'"""
    args = get_args()
    logger.debug(
        "Service: '%s', Game: '%s', Config Path: '%s'",
        args.service,
        args.game,
        args.config_path,
    )

    config_path = Path.cwd().resolve() / args.config_path
    try:
        config = load_config(config_path=config_path)
    except Exception as e:
        logger.error(e)
        raise e
    
    using_cuda: bool = (
        "cuda" in DEVICE if isinstance(DEVICE, str) else "cuda" in DEVICE.type
    )
    set_random_seed(config.seed, using_cuda=using_cuda)

    try:
        env = make_vec_env(
            env_id=lambda: make_env(args.game),
            n_envs=args.n_envs if args.n_envs is not None else 1,
            seed=config.seed,
            vec_env_cls=SubprocVecEnv
        )
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
    except FileNotFoundError as e:
        logger.error(e)
        raise e


    try:
        config.set_callback()
        agent = load_model(
            model_type=config.model_type,
            env=env,
            settings_config=config.model_settings,
            model_path=config.model_path,
        )
        call_service(
            service_name=args.service,
            agent=agent,
            config=config
        )

    except AttributeError as e:
        logger.error(e)
        raise

    finally:
        env.close()


def make_env(env_id: str) -> ...:
    """
    Make an RL training Env.

    Args:
        env (str): Name of game to build env with.
        Game must be implemented in stable_retro.

    Returns:
        RetroEnv: An RL env of the input game.
    """
    try:
        env = make(env_id)
    except FileNotFoundError:
        env = make(GAME_NAME_MAP[env_id])
    
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84,84))
    env = NormalizeReward(env)

    return env


def load_config(config_path: Path) -> ConfigData:
    """
    Load a YAML config file.

    Args:
        config_path (Path): Path to config file (must be .yaml or .yml file)

    Raises:
        FileNotFoundError: on config_path.is_file() returns False
        ValueError: on config_path.suffix not ".yaml" or ".yml"

    Returns:
        ConfigData: Config data model
    """
    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError(
            "Expected config suffix to be '.yaml' or '.yml', "
            f"received {config_path.suffix})"
        )

    try:
        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        raise

    try:
        individual_service_settings: dict[str, dict[str, Any]] = {
            k: data.pop(f"{k}_settings")
            for k in VALID_SERVICES
            if f"{k}_settings" in data.keys()
        }

        if "service_settings" in data.keys():
            data["service_settings"].update(individual_service_settings)
        else:
            data["service_settings"] = individual_service_settings

        return ConfigData(config_path=config_path, **data)

    except TypeError:
        logger.error(
            "Config data contained invalid field(s) and/or value(s): %s",
            str(list(data.items())),
        )
        raise


if __name__ == "__main__":
    main()
