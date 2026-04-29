from datetime import datetime
from pathlib import Path

import torch as th

import retro_rl_agents.rl_models as rl_models
import retro_rl_agents.services as services

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
GAME_NAME_MAP: dict[str, str] = {
    "sf2": "StreetFighterIISpecialChampionEdition-Genesis-v0"
}
VALID_GAMES: list[str] = list(GAME_NAME_MAP.keys()) + list(
    GAME_NAME_MAP.values()
)

serv_mods = next(Path(services.__path__[0]).walk(), (None, None, []))[2]
VALID_SERVICES: list[str] = [
    service_type.replace(".py", "")
    for service_type in serv_mods
    if service_type != "call.py" and service_type.endswith(".py")
]

rlm_mods = next(Path(rl_models.__path__[0]).walk(), (None, None, []))[2]
VALID_MODELS: list[str] = [
    model_type.replace(".py", "")
    for model_type in rlm_mods
    if model_type != "load.py" and model_type.endswith(".py")
]

LOG_DIR = Path.cwd().resolve() / ".logs" / datetime.now().strftime("%Y-%m-%d")
