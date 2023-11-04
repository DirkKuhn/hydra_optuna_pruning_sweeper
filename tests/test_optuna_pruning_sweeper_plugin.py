import os
import sys
from pathlib import Path
from typing import Any, List

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.test_utils import (
    TSweepRunner,
    chdir_plugin_root,
    run_process,
    run_python_script,
)
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    IntDistribution,
    FloatDistribution
)
from pytest import mark

from hydra_plugins.hydra_optuna_pruning_sweeper import _impl
from hydra_plugins.hydra_optuna_pruning_sweeper.optuna_pruning_sweeper import OptunaPruningSweeper

chdir_plugin_root()


def test_discovery() -> None:
    assert OptunaPruningSweeper.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ]


def check_distribution(expected: BaseDistribution, actual: BaseDistribution) -> None:
    if not isinstance(expected, CategoricalDistribution):
        assert expected == actual
        return

    assert isinstance(actual, CategoricalDistribution)
    # shuffle() will randomize the order of items in choices.
    assert set(expected.choices) == set(actual.choices)


@mark.parametrize(
    "input, expected",
    [
        ("key=choice(1,2)", CategoricalDistribution([1, 2])),
        ("key=choice(true, false)", CategoricalDistribution([True, False])),
        ("key=choice('hello', 'world')", CategoricalDistribution(["hello", "world"])),
        ("key=shuffle(range(1,3))", CategoricalDistribution((1, 2))),
        ("key=range(1,3)", IntDistribution(1, 3)),
        ("key=interval(1, 5)", FloatDistribution(1, 5)),
        ("key=int(interval(1, 5))", IntDistribution(1, 5)),
        ("key=tag(log, interval(1, 5))", FloatDistribution(1, 5, log=True)),
        ("key=tag(log, int(interval(1, 5)))", IntDistribution(1, 5, log=True)),
        ("key=range(0.5, 5.5, step=1)", FloatDistribution(0.5, 5.5, step=1)),
    ],
)
def test_create_optuna_distribution_from_override(input: str, expected: BaseDistribution) -> None:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides([input])[0]
    actual = _impl.create_optuna_distribution_from_override(parsed)
    check_distribution(expected, actual)


@mark.parametrize(
    "input, expected",
    [
        (["key=choice(1,2)"], ({"key": CategoricalDistribution([1, 2])}, {}, {})),
        (["key=5"], ({}, {"key": "5"}, {})),
        (
            ["key1=choice(1,2)", "key2=5"],
            ({"key1": CategoricalDistribution([1, 2])}, {"key2": "5"}, {}),
        ),
        (
            ["key1=choice(1,2)", "key2=5", "key3=range(1,3)"],
            (
                {
                    "key1": CategoricalDistribution([1, 2]),
                    "key3": IntDistribution(1, 3),
                },
                {"key2": "5"}, {},
            ),
        ),
        (['key=tag("1", choice(1,2))'], ({"key": CategoricalDistribution([1, 2])}, {}, {"key": [1]})),
        (['key=tag("false", choice(false,true))'], ({"key": CategoricalDistribution([False, True])}, {}, {"key": [False]})),
        (["key=tag(0:1, choice(1,2))"], ({"key": CategoricalDistribution([1, 2])}, {}, {"key": [1]})),
        (["key=tag(0:1, 1:2, choice(1,2))"], ({"key": CategoricalDistribution([1, 2])}, {}, {"key": [1, 2]})),
        (
            ["key1=tag(0:1, 1:2, choice(1,2))", "key2=5", "key3=range(1,3)"],
            (
                {
                    "key1": CategoricalDistribution([1, 2]),
                    "key3": IntDistribution(1, 3),
                },
                {"key2": "5"},
                {"key1": [1, 2],},
            ),
        ),
    ],
)
def test_create_params_and_manual_values(input: List[str], expected: Any) -> None:
    actual = _impl.create_params_and_manual_values(input, custom_search_space=[])
    assert actual == expected


def test_launch_jobs(hydra_sweep_runner: TSweepRunner) -> None:
    sweep = hydra_sweep_runner(
        calling_file=None,
        calling_module="hydra.test_utils.a_module",
        config_path="configs",
        config_name="compose.yaml",
        task_function=None,
        overrides=[
            "hydra/sweeper=OptunaPruningSweeper",
            "hydra/launcher=basic",
            "hydra.sweeper.n_trials=8",
            "hydra.sweeper.n_jobs=3",
        ],
    )
    with sweep:
        assert sweep.returns is None


@mark.parametrize("with_commandline", (True, False))
def test_optuna_example(with_commandline: bool, tmpdir: Path) -> None:
    storage = "sqlite:///" + os.path.join(str(tmpdir), "test.db")
    study_name = "test-optuna-example"
    cmd = [
        "examples/sphere/objective.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.storage={storage}",
        f"hydra.sweeper.study_name={study_name}",
        "hydra/sweeper/sampler=TPESampler",
        "hydra.sweeper.sampler.seed=123",
        "~z",
    ]
    if with_commandline:
        cmd += [
            "x=choice(0, 1, 2)",
            "y=0",  # Fixed parameter
        ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    study = optuna.load_study(storage=storage, study_name=study_name)
    best_trial = study.best_trial
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert returns["best_params"]["x"] == best_trial.params["x"]
    if with_commandline:
        assert "y" not in returns["best_params"]
        assert "y" not in best_trial.params
    else:
        assert returns["best_params"]["y"] == best_trial.params["y"]
    assert returns["best_value"] == best_trial.value
    # Check the search performance of the TPE sampler.
    # The threshold is the 95th percentile calculated with 1000 different seed values
    # to make the test robust against the detailed implementation of the sampler.
    # See https://github.com/facebookresearch/hydra/pull/1746#discussion_r681549830.
    assert returns["best_value"] <= 2.27


@mark.parametrize("num_trials", (10, 1))
def test_example_with_grid_sampler(
    tmpdir: Path,
    num_trials: int,
) -> None:
    storage = "sqlite:///" + os.path.join(str(tmpdir), "test.db")
    study_name = "test-grid-sampler"
    cmd = [
        "examples/sphere/objective.py",
        "--multirun",
        "--config-dir=tests/conf",
        "--config-name=test_grid",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=False",
        f"hydra.sweeper.n_trials={num_trials}",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.storage={storage}",
        f"hydra.sweeper.study_name={study_name}",
    ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    bv, bx, by, bz = (
        returns["best_value"],
        returns["best_params"]["x"],
        returns["best_params"]["y"],
        returns["best_params"]["z"],
    )
    if num_trials >= 12:
        assert bv == 1 and abs(bx) == 1 and by == 0
    else:
        assert bx in [-1, 1] and by in [-1, 0]
    assert bz in ["foo", "bar"]


@mark.parametrize("with_commandline", (True, False))
def test_optuna_multi_objective_example(with_commandline: bool, tmpdir: Path) -> None:
    cmd = [
        "examples/multi_objective/objective.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=RandomSampler",
        "hydra.sweeper.sampler.seed=123",
    ]
    if with_commandline:
        cmd += [
            "x=range(0, 5)",
            "y=range(0, 3)",
        ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    if with_commandline:
        for trial_x in returns["solutions"]:
            assert trial_x["params"]["x"] % 1 == 0
            assert trial_x["params"]["y"] % 1 == 0
            # The trials must not dominate each other.
            for trial_y in returns["solutions"]:
                assert not _dominates(trial_x, trial_y)
    else:
        for trial_x in returns["solutions"]:
            assert trial_x["params"]["x"] % 1 in {0, 0.5}
            assert trial_x["params"]["y"] % 1 in {0, 0.5}
            # The trials must not dominate each other.
            for trial_y in returns["solutions"]:
                assert not _dominates(trial_x, trial_y)


def _dominates(values_x: List[float], values_y: List[float]) -> bool:
    return all(x <= y for x, y in zip(values_x, values_y)) and any(
        x < y for x, y in zip(values_x, values_y)
    )


def test_optuna_custom_search_space_example(tmpdir: Path) -> None:
    max_z_difference_from_x = 0.3
    cmd = [
        "examples/custom_search_space/objective.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=RandomSampler",
        "hydra.sweeper.sampler.seed=123",
        f"max_z_difference_from_x={max_z_difference_from_x}",
    ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert (
        abs(returns["best_params"]["x"] - returns["best_params"]["z"])
        <= max_z_difference_from_x
    )
    w = returns["best_params"]["+w"]
    assert 0 <= w <= 1


def test_failure(tmpdir: Path) -> None:
    cmd = [
        sys.executable,
        "examples/sphere/objective.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=2",
        "hydra/sweeper/sampler=RandomSampler",
        "hydra.sweeper.sampler.seed=123",
        "error=true",
    ]
    out, err = run_process(cmd, print_error=False, raise_exception=False)
    error_string = "RuntimeError: cfg.error is True"
    assert error_string in err
