from typing import (
        Callable,
        List,
        Optional,
        Sequence,
        Iterable,
        Union,
        Literal,
        Mapping,
        Type,
        TYPE_CHECKING
    )

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

from hydra_plugins.custom_search_space import CustomSearchSpace

if TYPE_CHECKING:
    import optuna
    from optuna import Study
    from optuna.samplers import BaseSampler, GridSampler
    from optuna.pruners import BasePruner
    from optuna.storages import BaseStorage
    from optuna.trial import FrozenTrial
    from optuna.study import StudyDirection
    from dask.distributed import Client


class OptunaPruningSweeper(Sweeper):
    """Class to interface with Optuna"""

    def __init__(
            self,
            sampler: Optional[Union[
                "BaseSampler",
                Callable[[Mapping[str, Sequence["optuna.samplers._grid.GridValueType"]]], "GridSampler"]
            ]] = None,
            pruner: Optional["BasePruner"] = None,
            direction: Union[
                Literal["minimize"], Literal["maximize"], "StudyDirection",
                List[Union[Literal["minimize"], Literal["maximize"], "StudyDirection"]]
            ] = "minimize",
            storage: Optional[Union[str, "BaseStorage"]] = None,
            study_name: Optional[str] = None,
            n_trials: Optional[int] = None,
            n_jobs: int = 1,
            params: Optional[DictConfig] = None,
            custom_search_space: Optional[Union[CustomSearchSpace, List[CustomSearchSpace]]] = None,
            timeout: Optional[float] = None,
            catch: Optional[Iterable[Type[Exception]] | Type[Exception]] = None,
            callbacks: Optional[List[Callable[["Study", "FrozenTrial"], None]]] = None,
            gc_after_trial: bool = False,
            show_progress_bar: bool = False,
            dask_client: Optional[Callable[[], "Client"]] = None
    ):
        from ._impl import OptunaPruningSweeperImpl

        self.sweeper = OptunaPruningSweeperImpl(
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            storage=storage,
            study_name=study_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            params=params,
            custom_search_space=custom_search_space,
            timeout=timeout,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
            dask_client=dask_client
        )

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
