import yaml

from pathlib import Path
from stable_retro import RetroEnv, make

from retro_rl_agents.cli.arguments import get_args
from retro_rl_agents.utils.constants import GAME_NAME_MAP
from retro_rl_agents.rl_models.load import load_model
from retro_rl_agents.services.call import call_service
from retro_rl_agents.data_models.config_data import ConfigData


def main():
    """Entry point for calling services like 'train' or 'eval'"""
    args = get_args()

    env = make_env(args.game)
    
    config_path = Path.cwd().resolve() / args.config_path
    config = load_config(config_path=config_path)

    agent = load_model(
        model_type=config.model_type,
        env=env,
        settings_config=config.model_settings,
        model_path=config.model_path
    )

    call_service(
        service_name=args.service,
        agent=agent,
        config=config
    )

    env.close()


def make_env(env: str) -> RetroEnv:
    """
    Make an RL training Env.

    Args:
        env (str): Name of game to build env with.
        Game must be implemented in stable_retro.
    
    Returns:
        RetroEnv: An RL env of the input game.
    """
    try:
        return make(env)
    except FileNotFoundError:
        return make(GAME_NAME_MAP[env])
    

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
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError(
            "Expected config suffix to be '.yaml' or '.yml', "
            f"received {config_path.suffix})"
        )
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return ConfigData(config_path=config_path, **data)


if __name__ == "__main__":
    main()
