from dataclasses import dataclass, field
from typing import Any
from warnings import warn

from stable_baselines3.common.callbacks import CallbackList

from retro_rl_agents.callbacks.callback_factory import CallbackFactory
from retro_rl_agents.callbacks.external_cbs import register_external_callbacks


@dataclass
class ServiceData:
    """
    Dataclass for holding info about the service being called.
    """

    service_name: str
    settings: dict[str, Any] = field(default_factory=dict)
    cb_factory: CallbackFactory = CallbackFactory()

    def set_callback(self, **kwargs) -> None:
        cb_list: list[dict[str, Any]] | None = self.settings.get("callback")
        if not cb_list:
            return

        cb_names = [cfg["type"] for cfg in cb_list]
        register_external_callbacks(
            cb_factory=self.cb_factory, callback_list=cb_names
        )

        if "sb3_checkpoint" in cb_names:
            # This shows a potential arg against breaking this into its own cls
            check_idx: int = cb_names.index("sb3_checkpoint")
            save_path = kwargs.get("save_path")
            n_envs = kwargs.get("n_envs")
            if not save_path or not n_envs:
                warn(
                    "Missing either save_path or n_envs:"
                    f" {save_path = !r}; {n_envs = !r};"
                    " Removing 'sb3_checkpoint' from callback list."
                )
                cb_list.pop(check_idx)
            else:
                cb_list[check_idx]["save_path"] = save_path / "checkpoints"
                cb_list[check_idx]["save_freq"] = max(
                    cb_list[check_idx]["save_freq"] // n_envs, 1
                )

        self.settings["callback"] = CallbackList(
            [self.cb_factory.get_callback(cfg) for cfg in cb_list]
        )
