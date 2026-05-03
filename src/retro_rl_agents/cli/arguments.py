from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from retro_rl_agents.utils.constants import VALID_SERVICES


def get_args() -> Namespace:
    parser = ArgumentParser(
        prog="retro-rl-agents",
        description="Train RL agents to play retro games.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "service",
        type=str,
        help="Service to call for RL agent.",
        choices=VALID_SERVICES,
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a yaml config.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
