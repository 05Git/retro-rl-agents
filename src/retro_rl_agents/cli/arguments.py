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
        "game", type=str, help="Name of game to train RL agent on."
    )

    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        help="Path to a yaml config.",
        required=False,
    )

    parser.add_argument(
        "--n_envs",
        "-e",
        type=int,
        required=False,
        help="Number of parallel envs to run.",
    )

    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = f"configs/{args.game}/{args.service}.yml"

    return args


if __name__ == "__main__":
    args = get_args()
