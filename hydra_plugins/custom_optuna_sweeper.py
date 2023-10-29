import functools
import logging
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
)

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from optuna import Study
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    IntDistribution,
    FloatDistribution
)
from optuna.trial import Trial
from optuna.study import MaxTrialsCallback
from optuna.integration import DaskStorage

from .config import SamplerConfig, Direction
import hydra_plugins.trial_provider

log = logging.getLogger(__name__)


class CustomOptunaSweeper(Sweeper):
    def __init__(
            self,
            sampler: SamplerConfig,
            direction: Any,
            storage: Optional[Any],
            study_name: Optional[str],
            n_trials: int,
            n_jobs: int,
            custom_search_space: Optional[str],
            params: Optional[DictConfig],
    ) -> None:
        self.sampler = sampler
        self.direction = direction
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.custom_search_space_extender: Optional[
            Callable[[DictConfig, Trial], None]
        ] = None
        if custom_search_space:
            self.custom_search_space_extender = get_method(custom_search_space)
        self.params = params

        self.search_space_distributions: Optional[Dict[str, BaseDistribution]] = None
        self.fixed_params: Optional[Dict[str, Any]] = None
        self.manual_values: Optional[Dict[str, Sequence[Any]]] = None
        self.num_directions: int = len(self.direction) \
            if isinstance(self.direction, MutableSequence) else 1

    def setup(
            self,
            *,
            hydra_context: HydraContext,
            task_function: TaskFunction,
            config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

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
        ) = create_params_from_overrides(params_conf)

        is_grid_sampler = (
            isinstance(self.sampler, functools.partial)
            and self.sampler.func == optuna.samplers.GridSampler
        )
        if is_grid_sampler:
            self._setup_grid_sampler()

        # Remove fixed parameters from Optuna search space.
        for param_name in self.fixed_params:
            if param_name in self.search_space_distributions:
                del self.search_space_distributions[param_name]

        directions = self._get_directions()

        if self.n_jobs == 1:
            study = self._setup_study(directions)
            study.optimize(func=self._run_trial, n_trials=self.n_trials)
            self._serialize_results(study, len(directions))

        else:
            from dask.distributed import Client, wait

            with Client(n_workers=self.n_jobs) as client:
                print(f"Dask dashboard is available at {client.dashboard_link}")
                self.storage = DaskStorage(storage=self.storage, client=client)
                study = self._setup_study(directions)
                futures = [
                    client.submit(
                        study.optimize,
                        self._run_trial,
                        callbacks=[MaxTrialsCallback(n_trials=self.n_trials, states=None)],
                        pure=False
                    ) for _ in range(self.n_jobs)
                ]
                wait(futures)
                self._serialize_results(study, len(directions))

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

    def _get_directions(self) -> List[str]:
        if isinstance(self.direction, MutableSequence):
            return [d.name if isinstance(d, Direction) else d for d in self.direction]
        elif isinstance(self.direction, str):
            return [self.direction]
        return [self.direction.name]

    def _setup_study(self, directions: List[str]) -> Study:
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            directions=directions,
            load_if_exists=True,
        )
        self._enqueue_manual_values(study)
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(self.sampler).__name__}")
        log.info(f"Directions: {directions}")
        return study

    def _enqueue_manual_values(self, study: Study) -> None:
        num_manual_values = max(len(s) for s in self.manual_values.values())
        for i in range(num_manual_values):
            params = dict()
            for n, mv in self.manual_values.items():
                if i < len(mv):
                    params[n] = mv[i]
            study.enqueue_trial(params, skip_if_exists=True)

    def _run_trial(self, trial: Trial) -> float | Sequence[float]:
        # Share trial with task_function
        hydra_plugins.trial_provider.trial = trial

        overrides = self._configure_trial(trial)
        [ret] = self.launcher.launch([overrides], initial_job_idx=trial.number)

        if self.num_directions == 1:
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
            if len(values) != self.num_directions:
                raise ValueError(
                    "The number of the values and the number of the objectives are"
                    f" mismatched. Expect {self.num_directions}, but actually {len(values)}."
                )
        return values

    def _configure_trial(self, trial: Trial) -> Sequence[str]:
        for param_name, distribution in self.search_space_distributions.items():
            assert type(param_name) is str
            trial._suggest(param_name, distribution)
        for param_name, value in self.fixed_params.items():
            trial.set_user_attr(param_name, value)

        if self.custom_search_space_extender:
            assert self.config is not None
            self.custom_search_space_extender(self.config, trial)

        overlap = trial.params.keys() & trial.user_attrs
        if len(overlap):
            raise ValueError(
                "Overlapping fixed parameters and search space parameters found!"
                f"Overlapping parameters: {list(overlap)}"
            )
        params = dict(trial.params)
        params.update(self.fixed_params)

        return tuple(f"{name}={val}" for name, val in params.items())

    def _serialize_results(self, study: Study, num_directions: int) -> None:
        results_to_serialize: Dict[str, Any]
        if num_directions < 2:
            best_trial = study.best_trial
            results_to_serialize = {
                "name": "optuna",
                "best_params": best_trial.params,
                "best_value": best_trial.value,
            }
            log.info(f"Best parameters: {best_trial.params}")
            log.info(f"Best value: {best_trial.value}")
        else:
            best_trials = study.best_trials
            pareto_front = [
                {"params": t.params, "values": t.values} for t in best_trials
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


def create_params_from_overrides(
        arguments: List[str],
) -> Tuple[Dict[str, BaseDistribution], Dict[str, Any], Dict[str, Sequence[Any]]]:
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
            manual_values[param_name] = _extract_manual_values_from_tags(override)
        else:
            fixed_params[param_name] = value
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


def _extract_manual_values_from_tags(override: Override) -> Sequence[Any]:
    assert override.is_sweep_override()
    manual_values = [t.split(":", maxsplit=1) for t in override.value().tags if t != "log"]
    manual_values = [(int(pos), val) for pos, val in manual_values]
    assert (
        set(range(len(manual_values))) == set(pos for pos, _ in manual_values)
    ), (f"Expected manual values for {override.get_key_element()} to be numbered from 0 "
        f"to {len(manual_values)-1} but got numbers {set(pos for pos, _ in manual_values)}.")

    def extract_value(val: str) -> Any:
        from ast import literal_eval
        try:
            mv = literal_eval(val)
        except ValueError:
            mv = val
        return mv

    from operator import itemgetter
    manual_values = [extract_value(val) for _, val in sorted(manual_values, key=itemgetter(0))]
    return manual_values


def _to_grid_sampler_choices(distribution: BaseDistribution) -> Any:
    if isinstance(distribution, CategoricalDistribution):
        return distribution.choices
    elif isinstance(distribution, (IntDistribution, FloatDistribution)):
        assert (
                distribution.step is not None
        ), "`step` of IntDistribution and FloatDistribution must be a positive number."
        n_items = (distribution.high - distribution.low) // distribution.step
        return [distribution.low + i * distribution.step for i in range(n_items)]
    else:
        raise ValueError("GridSampler only supports discrete distributions.")
