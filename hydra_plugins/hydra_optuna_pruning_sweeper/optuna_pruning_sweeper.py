from typing import (
        Callable,
        List,
        Optional,
        Sequence,
        Iterable,
        Union,
        Mapping,
        Type,
        Container,
        TYPE_CHECKING
    )

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

from hydra_plugins.hydra_optuna_pruning_sweeper import CustomSearchSpace

if TYPE_CHECKING:
    import optuna
    from optuna import Study
    from optuna.samplers import BaseSampler, GridSampler
    from optuna.pruners import BasePruner
    from optuna.storages import BaseStorage
    from optuna.trial import FrozenTrial, TrialState
    from dask.distributed import Client
    from ._impl import DirectionType


class OptunaPruningSweeper(Sweeper):
    def __init__(
            self,
            sampler: Optional[Union[
                "BaseSampler",
                Callable[[Mapping[str, Sequence["optuna.samplers._grid.GridValueType"]]], "GridSampler"]
            ]] = None,
            pruner: Optional["BasePruner"] = None,
            direction: Union["DirectionType", List["DirectionType"]] = "minimize",
            storage: Optional[Union[str, "BaseStorage"]] = None,
            study_name: Optional[str] = None,
            n_trials: Optional[int] = None,
            n_trials_states: Optional[Container["TrialState"]] = None,
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
        """
        Class to interface with Optuna. This sweeper is used when ``override hydra/sweeper: OptunaPruningSweeper``
        is configured in a global defaults list or ``hydra/sweeper=OptunaPruningSweeper`` in the command line.
        This class and all arguments are recursively instantiated by hydra, which internally calls
        hydra.utils.instantiate (see https://hydra.cc/docs/advanced/instantiate_objects/overview/).

        :param sampler: Optuna sampler object. All current samplers are already configured with their default values.
            Add for example ``hydra/sweeper/sampler: TPESampler`` to configure the ``TPESampler``. Note that the
            ``GridSampler`` as an exception should be only partially instantiated
            (see https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation). Its argument
            ``search_space`` is provided by the ``OptunaPruningSweeper``. By default optuna's default sampler is used.
        :param pruner: Optuna pruner object. All current pruners are already configured with their default values.
            Add for example ``hydra/sweeper/sampler: HyperbandPruner`` to configure the ``HyperbandPruner``. By
            default optuna's default pruner is used.
        :param direction: Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
            maximization. You can also pass the corresponding ``optuna.study.StudyDirection`` object. During
            multi-objective optimization pass a sequence of directions. Optuna does not support pruning in the
            multi-objective case.
        :param storage: Database URL. If this argument is set to None, in-memory storage is used, and the ``Study``
            will not be persistent. When a database URL is passed, optuna internally uses ``SQLAlchemy`` to handle
            the database. If ``n_jobs>1`` this storage is wrapped in a ``optuna.storage.DaskStorage``.
        :param study_name: Study's name. If this argument is set to None, a unique name is generated automatically.
        :param n_trials: The total number of trials that will be run among all processes/workers. This argument together
            with ``n_trials_states`` is passed to a ``optuna.study.MaxTrialsCallback``. ``None`` represents no limit
            in terms of the number of trials. The study continues to create trials until the number of trials reaches
            ``n_trials``, ``timeout`` period elapses, ``stop()`` is called, or a termination signal such as SIGTERM or
            Ctrl+C is received.
        :param n_trials_states: Tuple of the ``optuna.trial.TrialState`` to be counted towards the max trials limit.
            Default value is ``(None,)`` which counts all states. This argument together with ``n_trials`` is passed to
            a ``optuna.study.MaxTrialsCallback``.
        :param n_jobs: If ``n_jobs>1`` the hyperparameter search is parallelized by wrapping the specified ``storage``
            in a ``optuna.storage.DaskStorage`` and using the passed ``dask_client``. In
        :param params: Map from parameter names to sweep overrides
            (https://hydra.cc/docs/advanced/override_grammar/extended/#sweeps). Use ``choice`` override instead of
            ``suggest_categorical`` (i.e. ``x: choice(false, true)``), ``range`` override instead of ``suggest_int``
            (i.e. ``x: range(1, 4)``) and ``interval`` override instead of ``suggest_float``
            (i.e. ``x: interval(1, 4)``). In case of ``range`` and ``interval`` add the tag "log" for a logarithmic
            search space (i.e. ``x: tag(log, range(1, 4))`` and ``x: tag(log, interval(1, 4))``).

            Manual values can be specified as tags, i.e. ``x: tag("1", log, interval(1, 4))``. Unfortunately, hydra
            throws an error if the tags can be converted to something over than a string. Therefore, numbers, booleans
            and ``null`` have to be surrounded by quotes. If multiple manual values are specified they have to be
            prefaced by their index, as hydra does not respect the order, i.e. ``x: tag(0:1, 1:3, log, interval(1, 4))``
            (first x=0 is tried then x=3 and later the values are sampled from [1, 4]). As hydra can't convert ``0:1``
            to a number, no quotes are required.
        :param custom_search_space: A single or a sequence of implementations of
            ``custom_search_space.CustomSearchSpace``. Used to dynamically create search spaces.
        :param timeout: Stop study after the given number of second(s). ``None`` represents no limit in terms of
            elapsed time. The study continues to create trials until the number of trials reaches, ``n_trials``,
            ``timeout`` period elapses, ``stop()`` is called or, a termination signal such as SIGTERM or Ctrl+C is
            received.
        :param catch: A study continues to run even when a trial raises one of the exceptions specified in this
            argument. Default is an empty tuple, i.e. the study will stop for any exception except for ``TrialPruned``.
        :param callbacks: List of callback functions that are invoked at the end of each trial. Each function must
            accept two parameters with the following types in this order: ``Study`` and ``FrozenTrial``.
            See the tutorial
            https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html#optuna-callback
            for how to use and implement callback functions.
        :param gc_after_trial: Flag to determine whether to automatically run garbage collection after each trial.
            Set to ``True`` to run the garbage collection, ``False`` otherwise. When it runs, it runs a full collection
            by internally calling ``gc.collect()``. If you see an increase in memory consumption over several trials,
            try setting this flag to ``True``.
        :param show_progress_bar: Flag to show progress bars or not. To disable progress bar, set this False.
            Currently, progress bar is experimental feature and disabled when n_trials is None, timeout is not None,
            and ``n_jobs>1``.
        :param dask_client: Callable which creates a ``dask.distributed.Client``. Libraries like
            https://jobqueue.dask.org/en/latest/ or
            https://mpi.dask.org/en/latest/ can be used to set up a dask cluster over multiple nodes.
            See ``examples/custom_search_space`` for an example on how to populate this argument.
        """
        from ._impl import OptunaPruningSweeperImpl

        self.sweeper = OptunaPruningSweeperImpl(
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            storage=storage,
            study_name=study_name,
            n_trials=n_trials,
            n_trials_states=n_trials_states,
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
