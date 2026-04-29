"""
Wrappers from external libs like Gymnasium
"""

import logging

from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    NormalizeReward,
)

from retro_rl_agents.env_wrappers.wrapper_factory import EnvWrapperFactory

logger = logging.getLogger(__name__)


_EXTERNAL_WRAPPER_REGISTRY = {
    "gym_grayscale": GrayscaleObservation,
    "gym_resizeobs": ResizeObservation,
    "gym_framestack": FrameStackObservation,
    "gym_normreward": NormalizeReward,
}


def register_external_env_wrappers(
    wrap_factory: EnvWrapperFactory, env_wrappers: list[str]
) -> None:
    for wrapper in env_wrappers:
        try:
            wrap_factory.register(
                key=wrapper, wrapper=_EXTERNAL_WRAPPER_REGISTRY[wrapper]
            )
        except KeyError:
            logger.error("Unexpected env wrapper: %s", wrapper.__repr__)
            raise
