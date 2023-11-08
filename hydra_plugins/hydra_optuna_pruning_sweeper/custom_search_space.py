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
    """
    Override this class to specify a dynamic search space with optional manual values.
    See https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
    """
    def manual_values(self) -> Dict[str, List[Any]]:
        """
        Override to specify manual values.

        :return: Map from hyperparameter name to a list of manual
            values which are sorted in the order in which they are tried.
            Each name should correspond to a value returned by ``suggest``.
        """
        return dict()

    @abstractmethod
    def suggest(self, cfg: DictConfig, trial: Trial) -> Dict[str, Any]:
        """
        Override to construct a dynamic search space using the ``optuna.trial.Trial`` object.

        :param cfg: Map which contains the output configuration assembled by hydra.
        :param trial: ``optuna.trial.Trial`` which contains the already suggested values under
            its property ``params`` and values of fixed params under its property ``user_attrs``.
        :return: Map from hyperparameter name to the next value to try.
        """


class ListSearchSpace(CustomSearchSpace):
    def __init__(
            self,
            name: str,
            min_entries: int, max_entries: int,
            min_value: float, max_value: float,
            use_float: bool, manual_values: Sequence[Any]
    ):
        """
        Example implementation of ``CustomSearchSpace`` which suggests a list of variable length.
        It can be used to suggest the architecture of a fully connected neural network.

        :param name: Name of the parameter for which a list of values should be suggested.
        :param min_entries: The minimum number of list entries.
        :param max_entries: The maximum number of list entries.
        :param min_value: The minimum value of each list entry.
        :param max_value: The maximum value of each list entry.
        :param use_float: Whether to suggest ints for floats for the list entries.
        :param manual_values: Sequence of fixed lists which should be
            tried before new values are suggested by the sampler.
        """
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
