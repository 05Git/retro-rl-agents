from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class CallbackFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        self._registry[name] = cls

    def get_callback(self, config: dict[str, Any]) -> BaseCallback:
        name: str = config.pop("type")
        if name is None:
            raise ValueError("'type' field not specified in callback config")
        try:
            return self._registry[name](**config)
        except KeyError:
            raise KeyError(f"Unknown Callback: {name!r}")
        
    def __repr__(self) -> str:
        return f"CallbackFactory({list(self._registry.keys())})"
