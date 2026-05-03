import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3.common.utils import set_random_seed

from retro_rl_agents.cli.arguments import get_args
from retro_rl_agents.domain_models.config_data import ConfigData
from retro_rl_agents.domain_models.env_data import EnvData
from retro_rl_agents.domain_models.service_data import ServiceData
from retro_rl_agents.rl_models.load import load_model
from retro_rl_agents.services.call import call_service
from retro_rl_agents.utils.constants import (
    DEVICE,
    LOG_DIR,
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
        "Service: '%s', Config Path: '%s'",
        args.service,
        args.config_path,
    )

    config_path: Path = Path.cwd().resolve() / args.config_path
    config_settings: dict[str, Any] = yaml.full_load(config_path.read_text())
    try:
        env_data = EnvData(**config_settings.pop("env", {}))
        env_data.set_wrappers()
        service_data = ServiceData(
            service_name=args.service,
            settings=config_settings.pop(args.service, {})
        )
        agent_config: dict[str, Any] = config_settings.pop("agent", {})
        agent_data = load_model(
            env                 = env_data.env,
            model_type          = agent_config.pop("model_type"),
            model_path          = agent_config.pop("model_path", None),
            settings_config     = agent_config.pop("model_settings", {}),
        )
        config_data = ConfigData(
            config_path         = config_path,
            agent_data          = agent_data,
            env_data            = env_data,
            service_data        = service_data,
            **config_settings
        )
        config_data.service_data.set_callback(
            save_path   = config_data.save_path,
            n_envs      = config_data.env_data.n_envs
        )
    except Exception as e:
        logger.error(e)
        raise e

    using_cuda: bool = (
        "cuda" in DEVICE if isinstance(DEVICE, str) else "cuda" in DEVICE.type
    )
    set_random_seed(config_data.env_data.seed, using_cuda=using_cuda)

    try:
        call_service(service_name=args.service, config=config_data)
    except Exception as e:
        logger.error(e)
        raise
    finally:
        env_data.env.close()


if __name__ == "__main__":
    main()
