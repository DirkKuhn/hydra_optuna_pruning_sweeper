# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Any, Sequence

import hydra
from omegaconf import DictConfig
from optuna.trial import Trial

from hydra_plugins.hydra_optuna_pruning_sweeper.custom_search_space import CustomSearchSpace


@hydra.main(version_base=None, config_path="conf", config_name="config")
def multi_dimensional_sphere(cfg: DictConfig) -> float:
    w: float = cfg.w
    x: float = cfg.x
    y: float = cfg.y
    z: float = cfg.z
    return w**2 + x**2 + y**2 + z**2


class MyCustomSearchSpace(CustomSearchSpace):
    z_name = "+z"
    w_name = "+w"

    def manual_values(self) -> Dict[str, Sequence[Any]]:
        return {self.z_name: [0.5, 1.5], self.w_name: [0, 1]}

    def suggest(self, cfg: DictConfig, trial: Trial) -> Dict[str, Any]:
        x_value = trial.params["+x"]

        z_value = trial.suggest_float(
            self.z_name,
            x_value - cfg.max_z_difference_from_x,
            x_value + cfg.max_z_difference_from_x,
        )
        w_value = trial.suggest_float("+w", 0.0, 1.0)  # note +w here, not w as w is a new parameter
        return {self.z_name: z_value, self.w_name: w_value}


if __name__ == "__main__":
    multi_dimensional_sphere()
