"""
Factory for producing env wrappers
"""

import gymnasium as gym


class EnvWrapperFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type[gym.Wrapper]] = {}

    def register(self, key: str, wrapper: type[gym.Wrapper]) -> None:
        self._registry[key] = wrapper

    def get_wrapper(self, wrapper_key: str) -> type[gym.Wrapper]:
        try:
            return self._registry[wrapper_key]
        except KeyError:
            raise KeyError(f"Unknown Env Wrapper: {wrapper_key!r}")

    def __repr__(self) -> str:
        return f"EnvWrapperFactory({list(self._registry.keys())})"
