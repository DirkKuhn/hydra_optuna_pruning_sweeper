import inspect

from hydra_zen import store

from .optuna_pruning_sweeper import OptunaPruningSweeper


store(
    OptunaPruningSweeper,
    group="hydra/sweeper",
    provider="optuna_pruning_sweeper"
)


def configure_optuna_samplers() -> None:
    from optuna import samplers

    optuna_samplers = [
        cls for name, cls in inspect.getmembers(samplers)
        if inspect.isclass(cls) and "Sampler" in name and "Base" not in name and name != "GridSampler"
    ]
    for sampler_cls in optuna_samplers:
        store(
            sampler_cls,
            group="hydra/sweeper/sampler",
            provider="optuna_pruning_sweeper"
        )
    store(
        samplers.GridSampler,
        # search_space will be populated at run time based on hydra.sweeper.params
        zen_partial=True,
        group="hydra/sweeper/sampler",
        provider="optuna_pruning_sweeper"
    )


def configure_optuna_pruners() -> None:
    from optuna import pruners

    optuna_pruners = [
        cls for name, cls in inspect.getmembers(pruners)
        if inspect.isclass(cls) and "Pruner" in name and "Base" not in name
    ]
    for pruner_cls in optuna_pruners:
        store(
            pruner_cls,
            group="hydra/sweeper/pruner",
            provider="optuna_pruning_sweeper"
        )


def configure_dask() -> None:
    from distributed import Client, LocalCluster, SpecCluster

    store(
        Client,
        group="hydra/sweeper/dask_client",
        # Partial as this should be instantiated in with statement.
        zen_partial=True,
        provider="optuna_pruning_sweeper"
    )
    store(
        LocalCluster,
        group="hydra/sweeper/dask_client/address",
        provider="optuna_pruning_sweeper"
    )
    store(
        SpecCluster,
        group="hydra/sweeper/dask_client/address",
        provider="optuna_pruning_sweeper"
    )


configure_optuna_samplers()
configure_optuna_pruners()
configure_dask()

store.add_to_hydra_store()
