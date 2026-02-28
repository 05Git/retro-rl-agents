from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path
from typing import Any
from datetime import datetime

def service(agent: BaseAlgorithm, config: dict[str, Any]) -> None:    
    train_settings: dict[str, Any] = config["train_settings"]

    save_dir: str = (
        config["save_dir"]
        if "save_dir"  in config.keys()
        else "trained_agents"
    )
    run_id = (
        config["run_id"]
        if "run_id" in config.keys()
        else datetime.now().isoformat(timespec="seconds")
    )
    save_path = Path(
        Path.cwd().resolve(),
        save_dir,
        config["model_type"],
        run_id
    )
    save_path.mkdir(parents=True, exist_ok=True)
    save_name = str(train_settings["total_timesteps"])

    agent.learn(**train_settings)
    agent.save(path=save_path / save_name)