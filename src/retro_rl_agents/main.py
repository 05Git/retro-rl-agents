import yaml

from pathlib import Path
from stable_retro import RetroEnv, make

from retro_rl_agents.cli.arguments import get_args
from retro_rl_agents.utils.constants import GAME_NAME_MAP
from retro_rl_agents.rl_models.load import load_model
from retro_rl_agents.services.call import call_service
from retro_rl_agents.data_models.config_data import ConfigData

def make_env(env: str) -> RetroEnv:
    try:
        return make(env)
    except FileNotFoundError:
        return make(GAME_NAME_MAP[env])
    
def load_config(config_path: Path) -> ConfigData:
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

def main():
    args = get_args()

    game_name = args.game
    env = make_env(game_name)
    
    service_type = args.service
    config_path = args.config_path

    if config_path is None:
        config_path = f"configs/{game_name}/{service_type}.yml"

    resolved_config_path = Path.cwd().resolve() / config_path
    config = load_config(config_path=resolved_config_path)
    
    agent = load_model(
        model_type=config.model_type,
        env=env,
        settings_config=config.model_settings,
        model_path=config.model_path
    )

    call_service(
        service_name=service_type,
        agent=agent,
        config=config
    )

    env.close()

if __name__ == "__main__":
    main()
