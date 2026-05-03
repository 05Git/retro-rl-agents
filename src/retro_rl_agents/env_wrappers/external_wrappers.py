"""
Wrappers from external libs like Gymnasium
"""

import logging

from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.normalize import NormalizeReward
from gymnasium.wrappers.resize_observation import ResizeObservation

from retro_rl_agents.env_wrappers.wrapper_factory import EnvWrapperFactory

logger = logging.getLogger(__name__)


_EXTERNAL_WRAPPER_REGISTRY = {
    "gym_grayscale": GrayScaleObservation,
    "gym_resizeobs": ResizeObservation,
    "gym_framestack": FrameStack,
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
