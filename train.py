import stable_retro as retro
from argparse import ArgumentParser, Namespace

from constants import GAME_NAME_MAP

def main():
    args = parse_args()
    game = GAME_NAME_MAP[args.game]
    env = retro.make(game=game)
    env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
            break
    env.close()

def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog="Train Retro RL Agents",
    )

    parser.add_argument(
        "game",
        type=str
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
