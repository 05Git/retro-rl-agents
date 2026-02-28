import stable_retro as retro
import yaml

from pathlib import Path
from typing import Any

from retro_rl_agents.cli.arguments import get_args
from retro_rl_agents.utils.constants import GAME_NAME_MAP
from retro_rl_agents.rl_models.load import load_model
from retro_rl_agents.services.call import call_service

def make_env(env: str) -> retro.RetroEnv:
    try:
        return retro.make(env)
    except FileNotFoundError:
        return retro.make(GAME_NAME_MAP[env])
    
def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError(
            "Expected config suffix to be '.yaml' or '.yml', "
            f"received {config_path.suffix})"
        )
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    args  = get_args()

    game_name = args.game
    env = make_env(game_name)
    
    service_type = args.service
    config_path = args.config_path
    if config_path is None:
        config_path = f"configs/{game_name}/{service_type}.yml"
    resolved_config_path = Path.cwd().resolve() / config_path
    config = load_config(config_path=resolved_config_path)

    try:
        model_type: str = config["model_type"]
    except KeyError:
        err_msg_args = [
            "No 'model_type' field detected from model_type config."
            "Please ensure the config has a 'model_type' field which"
            "specifies which type of model_type you want to use."
        ]
        raise KeyError(" ".join(err_msg_args))
    
    model_path = config.get("model_path")
    if model_path is not None:
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    settings_config = config.get("model_settings", {})
    agent = load_model(
        model_type=model_type,
        env=env,
        settings_config=settings_config,
        model_path=model_path
    )

    call_service(
        service_name=service_type,
        agent=agent,
        service_config=config
    )

    env.close()

if __name__ == "__main__":
    main()
