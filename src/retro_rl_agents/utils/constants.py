import torch as th
from pathlib import Path

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
GAME_NAME_MAP: dict[str, str] = {
    "sf2": "StreetFighterIISpecialChampionEdition-Genesis-v0"
}
VALID_SERVICES: list[str] = [
    "train"
]
LOG_DIR = Path.cwd().resolve() / ".logs"