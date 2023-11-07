from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Sequence
)

from omegaconf import DictConfig
from optuna.trial import Trial


class CustomSearchSpace(ABC):
    def manual_values(self) -> Dict[str, List[Any]]:
        return dict()

    @abstractmethod
    def suggest(self, cfg: DictConfig, trial: Trial) -> Dict[str, Any]:
        pass


class ListSearchSpace(CustomSearchSpace):
    def __init__(
            self,
            name: str,
            min_entries: int, max_entries: int,
            min_value: float, max_value: float,
            use_float: bool, manual_values: Sequence[Any]
    ):
        assert min_entries < max_entries, f"``min_entries`` should be lower than ``max_entries``!"
        assert min_value < max_value, f"``min_value`` should be lower than ``max_value``!"
        self.name = name
        self.min_entries = min_entries
        self.max_entries = max_entries
        self.min_value = min_value
        self.max_value = max_value
        self._manual_values = [list(mv) for mv in manual_values]
        self.use_float = use_float

    def manual_values(self) -> Dict[str, List[Any]]:
        return {self.name: self._manual_values}

    def suggest(self, cfg: DictConfig, trial: Trial) -> Dict[str, Any]:
        num_entries = trial.suggest_int(
            f"{self.name}.num_entries", low=self.min_entries, high=self.max_entries
        )
        suggest = trial.suggest_float if self.use_float else trial.suggest_int
        values = [
            suggest(name=f"{self.name}.{i}", low=self.min_value, high=self.max_value)
            for i in range(num_entries)
        ]
        return {self.name: values}
