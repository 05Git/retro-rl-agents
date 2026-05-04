"""
Environment model which wraps the given training env
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from retro_rl_agents.env_wrappers.external_wrappers import (
    register_external_env_wrappers,
)
from retro_rl_agents.env_wrappers.wrapper_factory import EnvWrapperFactory
from retro_rl_agents.utils.constants import GAME_NAME_MAP, VALID_GAMES


@dataclass
class EnvData:
    env_name: str
    n_envs: int | None
    seed: int = 0
    env_wrappers: list[dict[str, Any]] = field(default_factory=list)
    venv_cls: type[SubprocVecEnv] | type[DummyVecEnv] | None = None
    wrap_factory: EnvWrapperFactory = EnvWrapperFactory()

    def __post_init__(self) -> None:
        if self.env_name not in VALID_GAMES:
            raise ValueError(
                "Invalid game specified. Valid games: "
                + str(list(GAME_NAME_MAP.keys()))
            )
        if self.env_name in GAME_NAME_MAP.keys():
            self.env_name = GAME_NAME_MAP[self.env_name]

        if isinstance(self.n_envs, int) and self.n_envs <= 0:
            raise ValueError(
                "Number of envs must be greater than 0. Amount chosen: "
                + str(self.n_envs)
            )

        if isinstance(self.venv_cls, str):
            match self.venv_cls.lower():
                case "subproc":
                    self.venv_cls = SubprocVecEnv
                case "dummy":
                    self.venv_cls = DummyVecEnv
                case _:
                    self.venv_cls = None

    @property
    def env(self) -> retro.RetroEnv | gym.Env | VecEnv:
        if self.venv_cls is not None:
            return self._vec_env()
        return self._make_env()

    def _make_env(self) -> retro.RetroEnv | gym.Env:
        env = retro.make(self.env_name)
        for cfg in deepcopy(self.env_wrappers):
            wrap_key: str = cfg.pop("type", "")
            wrapper = self.wrap_factory.get_wrapper(wrap_key)
            env = wrapper(env, **cfg)
        return env

    def _vec_env(self) -> VecEnv:
        if self.venv_cls is None:
            raise ValueError(
                "VecEnv Class set to 'none', expected 'subproc' or 'dummy'."
            )
        if self.n_envs is None:
            raise ValueError(
                "'n_envs' is None, must set 'n_envs' >= 1 for vecenv."
            )
        return make_vec_env(
            env_id=lambda: self._make_env(),
            n_envs=self.n_envs,
            seed=self.seed,
            vec_env_cls=self.venv_cls,
        )

    def set_wrappers(self) -> None:
        wrapper_names: list[str] = [
            n for c in self.env_wrappers if (n := c.get("type"))
        ]
        register_external_env_wrappers(
            wrap_factory=self.wrap_factory, env_wrappers=wrapper_names
        )

    @property
    def serializable_env_settings(self) -> dict[str, Any]:
        serializable_settings: dict[str, Any] = {
            "seed": self.seed,
            "n_envs": self.n_envs,
            "venv_cls": repr(self.venv_cls),
            "env_wrappers": self.env_wrappers,
            "wrapper_factory": repr(self.wrap_factory),
        }
        return serializable_settings
