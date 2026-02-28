from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

# from .validate import validate_args
from retro_rl_agents.utils.constants import VALID_SERVICES

def get_args() -> Namespace:
    parser = ArgumentParser(
        prog="retro-agents",
        description="Train RL agents to play retro games.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "service",
        type=str,
        help="Service to call for RL agent.",
        choices=VALID_SERVICES
    )

    parser.add_argument(
        "game",
        type=str,
        help="Name of game to train RL agent on."
    )

    parser.add_argument(
        "--config-path", "-c",
        type=str,
        help="Path to a yaml config",
        required=False
    )

    args = parser.parse_args()
    # validate_args(args)

    return args

if __name__ == "__main__":
    get_args()