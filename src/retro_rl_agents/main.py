import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3.common.utils import set_random_seed

from retro_rl_agents.cli.arguments import get_args
from retro_rl_agents.domain_models.config_data import ConfigData
from retro_rl_agents.domain_models.env_model import EnvModel
from retro_rl_agents.rl_models.load import load_model
from retro_rl_agents.services.call import call_service
from retro_rl_agents.utils.constants import (
    DEVICE,
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

    config_path: Path = Path.cwd().resolve() / args.config_path
    config_data: dict[str, Any] = yaml.full_load(config_path.read_text())
    try:
        env_config: dict[str, Any] = config_data.pop("env_model", {})
        config: ConfigData = load_config(
            config_path=config_path, n_envs=args.n_envs, config_cfg=config_data
        )
        env_config["seed"] = config.seed
        if not args.n_envs and env_config["venv_cls"] is not None:
            logger.warning(
                "'venv_cls' is set to %s, but no arg was given"
                " for the number of parallel envs. Setting 'venv_cls'"
                " to 'None'.",
                env_config["venv_cls"]
            )
            env_config["venv_cls"] = None
        env_model: EnvModel = make_env_model(
            env_name=args.game, n_envs=args.n_envs, env_cfg=env_config
        )
    except Exception as e:
        logger.error(e)
        raise e

    using_cuda: bool = (
        "cuda" in DEVICE if isinstance(DEVICE, str) else "cuda" in DEVICE.type
    )
    set_random_seed(config.seed, using_cuda=using_cuda)

    try:
        config.set_callback()
        env_model.set_wrappers()
        agent = load_model(
            model_type=config.model_type,
            env=env_model.env,
            settings_config=config.model_settings,
            model_path=config.model_path,
        )
        call_service(service_name=args.service, agent=agent, config=config)

    except AttributeError as e:
        logger.error(e)
        raise

    finally:
        env_model.env.close()


def make_env_model(
    env_name: str, n_envs: int | None, env_cfg: dict[str, Any] = {}
) -> EnvModel:
    try:
        return EnvModel(env_name=env_name, n_envs=n_envs, **env_cfg)
    except ValueError as e:
        logger.error("Unable to load Env Model: %s", e)
        raise e


def load_config(
    config_path: Path, n_envs: int | None, config_cfg: dict[str, Any] = {}
) -> ConfigData:
    try:
        individual_service_settings: dict[str, dict[str, Any]] = {
            k: config_cfg.pop(serv_settings)
            for k in VALID_SERVICES
            if (serv_settings := f"{k}_settings") in config_cfg.keys()
        }

        if "service_settings" in config_cfg.keys():
            config_cfg["service_settings"].update(individual_service_settings)
        else:
            config_cfg["service_settings"] = individual_service_settings

        if isinstance(n_envs, int) and n_envs <= 0:
            raise ValueError(
                "Number of envs must be greater than 0."
                f" n_envs currently set to: {n_envs}"
            )
        config_cfg["n_envs"] = n_envs

        return ConfigData(config_path=config_path, **config_cfg)

    except TypeError:
        logger.error(
            "Config config_cfg contained invalid field(s) and/or value(s): %s",
            str(list(config_cfg.items())),
        )
        raise


if __name__ == "__main__":
    main()
