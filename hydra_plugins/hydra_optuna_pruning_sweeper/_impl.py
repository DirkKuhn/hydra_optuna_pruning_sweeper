import functools
import logging
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    MutableSequence,
    Optional,
    Sequence,
    Iterable,
    Union,
    Literal,
    Mapping,
    Type,
    Container
)
from operator import itemgetter
from pathlib import Path

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer
)
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf
from optuna import Study
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    IntDistribution,
    FloatDistribution
)
from optuna.samplers import BaseSampler, GridSampler
from optuna.pruners import BasePruner
from optuna.storages import BaseStorage
from optuna.trial import Trial, FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback, StudyDirection
from optuna.integration import DaskStorage
from dask.distributed import Client, wait

from hydra_plugins.hydra_optuna_pruning_sweeper import trial_provider
from hydra_plugins.hydra_optuna_pruning_sweeper import CustomSearchSpace


log = logging.getLogger(__name__)


DirectionType = Union[Literal["minimize"], Literal["maximize"], StudyDirection]


class OptunaPruningSweeperImpl(Sweeper):
    def __init__(
            self,
            sampler: Optional[Union[
                BaseSampler,
                Callable[[Mapping[str, Sequence[optuna.samplers._grid.GridValueType]]], GridSampler]
            ]],
            pruner: Optional[BasePruner],
            direction: Union[DirectionType, List[DirectionType]],
            storage: Optional[Union[str, BaseStorage]],
            study_name: Optional[str],
            n_trials: Optional[int],
            n_trials_states: Optional[Container[TrialState]],
            n_jobs: int,
            params: Optional[DictConfig],
            custom_search_space: Optional[Union[CustomSearchSpace, List[CustomSearchSpace]]],
            timeout: Optional[float],
            catch: Optional[Iterable[Type[Exception]] | Type[Exception]],
            callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]],
            gc_after_trial: bool,
            show_progress_bar: bool,
            dask_client: Optional[Callable[[], Client]]
    ):
        if n_jobs == 1 and dask_client:
            warnings.warn(
                "As ``n_jobs=1`` dask is not used. Specify ``n_jobs>1`` to use dask"
            )

        self.sampler = sampler
        self.directions = direction if isinstance(direction, MutableSequence) else [direction]
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_trials_states = n_trials_states
        self.n_jobs = n_jobs
        self.params = params
        if custom_search_space is None:
            self.custom_search_space = []
        elif isinstance(custom_search_space, Sequence):
            self.custom_search_space = custom_search_space
        else:
            self.custom_search_space = [custom_search_space]

        self.pruner = pruner
        self.timeout = timeout
        self.catch = () if catch is not None else catch
        self.callbacks = callbacks if callbacks else []
        self.gc_after_trial = gc_after_trial
        self.show_progress_bar = show_progress_bar

        self.dask_client = dask_client

        self.sweep_dir: Optional[str] = None
        self.search_space_distributions: Optional[Dict[str, BaseDistribution]] = None
        self.fixed_params: Optional[Dict[str, Any]] = None
        self.manual_values: Optional[Dict[str, List[Any]]] = None

    def setup(
            self,
            *,
            hydra_context: HydraContext,
            task_function: TaskFunction,
            config: DictConfig
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir
        # Already create directory, in case the study is placed there
        Path(str(self.sweep_dir)).mkdir(parents=True, exist_ok=True)

        # For some reason `isinstance` does not work
        assert self.launcher.__class__.__name__ == 'BasicLauncher', (
            f"This plugin assumes that the ``BasicLauncher`` (the default launcher) is used, "
            f"but got {self.launcher}. If you want to parallelize the hyperparameter search "
            f"simply set ``n_jobs>1``, then dask is used automatically. If you still need "
            f"another launcher, use the original optuna sweeper."
        )

        # Setup optuna logging
        optuna.logging.enable_propagation()       # Propagate logs to the root logger
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        if self.params is None:
            self.params = OmegaConf.create({})

        params_conf = self._parse_sweeper_params_config()
        params_conf.extend(arguments)

        (
            self.search_space_distributions,
            self.fixed_params,
            self.manual_values
        ) = create_params_and_manual_values(params_conf, self.custom_search_space)

        is_grid_sampler = (
            isinstance(self.sampler, functools.partial)
            and self.sampler.func == optuna.samplers.GridSampler
        )
        if is_grid_sampler:
            self._setup_grid_sampler()

        self._remove_fixed_params_from_search_space()

        if self.n_jobs == 1:
            self._run_sequential()
        else:
            self._run_parallel()

    def _parse_sweeper_params_config(self) -> List[str]:
        if not self.params:
            return []

        return [f"{k!s}={v}" for k, v in self.params.items()]

    def _setup_grid_sampler(self):
        search_space_for_grid_sampler = {
            name: _to_grid_sampler_choices(distribution)
            for name, distribution in self.search_space_distributions.items()
        }
        self.sampler = self.sampler(search_space_for_grid_sampler)
        n_trial = 1
        for v in search_space_for_grid_sampler.values():
            n_trial *= len(v)
        self.n_trials = min(self.n_trials, n_trial)
        log.info(
            f"Updating num of trials to {self.n_trials} due to using GridSampler."
        )

    def _remove_fixed_params_from_search_space(self):
        for param_name in self.fixed_params:
            if param_name in self.search_space_distributions:
                del self.search_space_distributions[param_name]

    def _run_sequential(self) -> None:
        study = self._setup_study()
        study.optimize(
            func=self._run_trial,
            timeout=self.timeout,
            catch=self.catch,
            callbacks=self._setup_callbacks(),
            gc_after_trial=self.gc_after_trial,
            show_progress_bar=self.show_progress_bar
        )
        self._serialize_results(study)

    def _run_parallel(self) -> None:
        with (
                self.dask_client() if self.dask_client else Client(n_workers=self.n_jobs)
        ) as client:
            print(f"Dask dashboard is available at {client.dashboard_link}")
            self.storage = DaskStorage(storage=self.storage, client=client)
            study = self._setup_study()
            futures = [
                client.submit(
                    study.optimize,
                    self._run_trial,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    catch=self.catch,
                    callbacks=self._setup_callbacks(),
                    gc_after_trial=self.gc_after_trial,
                    show_progress_bar=self.show_progress_bar,
                    pure=False
                ) for _ in range(self.n_jobs)
            ]
            wait(futures)
            self._serialize_results(study)

    def _setup_study(self) -> Study:
        study = optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name,
            directions=self.directions,
            load_if_exists=True,
        )
        self._enqueue_manual_values(study)
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(study.sampler).__name__}")
        log.info(f"Pruner: {type(study.pruner).__name__}")
        log.info(f"Directions: {self.directions}")
        return study

    def _enqueue_manual_values(self, study: Study) -> None:
        if not self.manual_values:
            return

        num_manual_values = max(len(s) for s in self.manual_values.values())
        for i in range(num_manual_values):
            params = dict()
            for n, mv in self.manual_values.items():
                if i < len(mv):
                    params[n] = mv[i]
            study.enqueue_trial(params, skip_if_exists=True)

    def _run_trial(self, trial: Trial) -> float | Sequence[float]:
        # Share trial with task_function
        trial_provider.trial = trial

        overrides = self._configure_trial(trial)
        [ret] = self.launcher.launch([overrides], initial_job_idx=trial.number)

        num_directions = len(self.directions)
        if num_directions == 1:
            try:
                values = [float(ret.return_value)]
            except (ValueError, TypeError):
                raise ValueError(
                    f"Return value must be float-castable. Got '{ret.return_value}'."
                ).with_traceback(sys.exc_info()[2])
        else:
            try:
                values = [float(v) for v in ret.return_value]
            except (ValueError, TypeError):
                raise ValueError(
                    "Return value must be a list or tuple of float-castable values."
                    f" Got '{ret.return_value}'."
                ).with_traceback(sys.exc_info()[2])
            if len(values) != num_directions:
                raise ValueError(
                    "The number of the values and the number of the objectives are"
                    f" mismatched. Expect {num_directions}, but actually {len(values)}."
                )
        return values

    def _configure_trial(self, trial: Trial) -> Sequence[str]:
        params = {
            name: trial._suggest(name, distribution)
            for name, distribution in self.search_space_distributions.items()
        }
        assert self.config is not None
        custom_params = [cs.suggest(self.config, trial) for cs in self.custom_search_space]
        for name, value in self.fixed_params.items():
            trial.set_user_attr(name, value)

        overlap = set.intersection(
            set(params.keys()), trial.user_attrs, *[d.keys() for d in custom_params]
        )
        if len(overlap):
            raise ValueError(
                "Overlapping fixed parameters and search space parameters found!"
                f"Overlapping parameters: {list(overlap)}"
            )

        for cp in custom_params:
            params.update(cp)
        params.update(self.fixed_params)

        return tuple(f"{name}={val}" for name, val in params.items())

    def _setup_callbacks(self) -> List[Callable[[Study, FrozenTrial], None]]:
        if self.n_trials is None:
            return self.callbacks
        else:
            return [
                MaxTrialsCallback(n_trials=self.n_trials, states=self.n_trials_states),
                *self.callbacks
            ]

    def _serialize_results(self, study: Study) -> None:
        results_to_serialize: Dict[str, Any]
        if len(self.directions) < 2:
            best_trial = study.best_trial
            results_to_serialize = {
                "name": "optuna",
                "trial_number": best_trial.number,
                "best_params": best_trial.params,
                "best_value": best_trial.value,
            }
            log.info(f"Best parameters: {best_trial.params}")
            log.info(f"Best value: {best_trial.value}")
        else:
            best_trials = study.best_trials
            pareto_front = [
                {"trial_number": t.number, "params": t.params, "values": t.values}
                for t in best_trials
            ]
            results_to_serialize = {
                "name": "optuna",
                "solutions": pareto_front,
            }
            log.info(f"Number of Pareto solutions: {len(best_trials)}")
            for t in best_trials:
                log.info(f"    Values: {t.values}, Params: {t.params}")
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )


def create_params_and_manual_values(
        arguments: List[str], custom_search_space: List[CustomSearchSpace]
) -> Tuple[Dict[str, BaseDistribution], Dict[str, Any], Dict[str, List[Any]]]:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(arguments)
    search_space_distributions = dict()
    fixed_params = dict()
    manual_values = dict()

    for override in parsed:
        param_name = override.get_key_element()
        value = create_optuna_distribution_from_override(override)
        if isinstance(value, BaseDistribution):
            search_space_distributions[param_name] = value
            mv = _extract_manual_values_from_tags(override)
            if mv:
                manual_values[param_name] = mv
        else:
            fixed_params[param_name] = value

    if bool(manual_values) and bool(custom_search_space):
        overlap = set.intersection(
            set(manual_values.keys()),
            *[cs.manual_values().keys() for cs in custom_search_space]
        )
        if len(overlap):
            raise ValueError(
                "Overlapping manual values found!"
                f"Overlapping manual values: {list(overlap)}"
            )
    for cs in custom_search_space:
        manual_values.update(cs.manual_values())

    return search_space_distributions, fixed_params, manual_values


def create_optuna_distribution_from_override(override: Override) -> Any:
    if not override.is_sweep_override():
        return override.get_value_element_as_str()

    value = override.value()
    choices: List[CategoricalChoiceType] = []
    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        for x in override.sweep_iterator(transformer=Transformer.encode):
            assert isinstance(
                x, (str, int, float, bool, type(None))
            ), f"A choice sweep expects str, int, float, bool, or None type. Got {type(x)}."
            choices.append(x)
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        assert isinstance(value, RangeSweep)
        assert value.start is not None
        assert value.stop is not None
        if value.shuffle:
            for x in override.sweep_iterator(transformer=Transformer.encode):
                assert isinstance(
                    x, (str, int, float, bool, type(None))
                ), f"A choice sweep expects str, int, float, bool, or None type. Got {type(x)}."
                choices.append(x)
            return CategoricalDistribution(choices)
        if (
                isinstance(value.start, float)
                or isinstance(value.stop, float)
                or isinstance(value.step, float)
        ):
            return FloatDistribution(value.start, value.stop, step=value.step)
        return IntDistribution(int(value.start), int(value.stop), step=int(value.step))

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        assert value.start is not None
        assert value.end is not None
        if "log" in value.tags:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntDistribution(int(value.start), int(value.end), log=True)
            return FloatDistribution(value.start, value.end, log=True)
        else:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntDistribution(value.start, value.end)
            return FloatDistribution(value.start, value.end)

    raise NotImplementedError(f"{override} is not supported by Optuna sweeper.")


def _extract_manual_values_from_tags(override: Override) -> Optional[List[Any]]:
    assert override.is_sweep_override()
    manual_values = [t for t in override.value().tags if t != "log"]
    if not manual_values:
        return None

    if len(manual_values) == 1 and ":" in manual_values[0] or len(manual_values) > 1:
        assert all(":" in t for t in manual_values), (
            'Preface each manual value with "idx:" where idx is its zero based index, '
            f'if there is more than just one manual value. Error for {override.get_key_element()}.'
        )
        manual_values = [t.split(":", maxsplit=1) for t in manual_values]
        manual_values = [(int(pos), val) for pos, val in manual_values]
        assert (
            set(range(len(manual_values))) == set(pos for pos, _ in manual_values)
        ), (f"Expected manual values for {override.get_key_element()} to be numbered from 0 "
            f"to {len(manual_values)-1} but got numbers {[pos for pos, _ in manual_values]}.")
        manual_values = [val for _, val in sorted(manual_values, key=itemgetter(0))]

    manual_values = [_extract_value(val) for val in manual_values]
    return manual_values


def _extract_value(val: str) -> Any:
    from ast import literal_eval

    if val == 'false':
        return False
    elif val == 'true':
        return True
    elif val == 'null':
        return None

    try:
        mv = literal_eval(val)
    except ValueError:
        mv = val
    return mv


def _to_grid_sampler_choices(distribution: BaseDistribution) -> Any:
    if isinstance(distribution, CategoricalDistribution):
        return distribution.choices
    elif isinstance(distribution, (IntDistribution, FloatDistribution)):
        assert (
                distribution.step is not None
        ), "`step` of IntDistribution and FloatDistribution must be a positive number."
        n_items = int((distribution.high - distribution.low) // distribution.step)
        return [distribution.low + i * distribution.step for i in range(n_items)]
    else:
        raise ValueError("GridSampler only supports discrete distributions.")
