import logging

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    EventCallback,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)

from retro_rl_agents.callbacks.callback_factory import CallbackFactory

logger = logging.getLogger(__name__)

_EXTERNAL_CALLBACK_REGISTRY: dict[str, type] = {
    "sb3_eval": EvalCallback,
    "sb3_checkpoint": CheckpointCallback,
    "sb3_event": EventCallback,
    "sb3_stop_on_max_eps": StopTrainingOnMaxEpisodes,
    "sb3_stop_on_no_improve": StopTrainingOnNoModelImprovement,
    "sb3_stop_on_rew_threshold": StopTrainingOnRewardThreshold,
}


def register_external_callbacks(
    cb_factory: CallbackFactory, callback_list: list[str]
) -> None:
    for callback in callback_list:
        try:
            cb_factory.register(callback, _EXTERNAL_CALLBACK_REGISTRY[callback])
        except KeyError:
            logger.error("Unexpected callback name: %s", callback.__repr__)
            raise
