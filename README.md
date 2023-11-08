# Optuna Sweeper plugin with Pruning

This plugin enables Hydra applications to utilize Optuna for the optimization of the parameters of experiments.
In contrast to the original [Optuna Sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/)
this plugin has the following advantages:

+ Pruning with
  ```python
    from hydra_plugins.hydra_optuna_pruning_sweeper import trial_provider
    ...
    trial = trial_provider.trial
    ...
    trial.report(score, step)
    if trial.should_prune():
        ...
    ...
  ```
  in the objective function.
+ Manual specification of hyperparameters which are used at the beginning of the hyperparameter search
  (see https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/008_specify_params.html).
+ Simple parallelization across processes with the
  [``optuna.integration.DaskStorage``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.DaskStorage.html#optuna.integration.DaskStorage)
  simply by specifying ``n_workers>1``.
+ Parallelization across nodes with the
  [``optuna.integration.DaskStorage``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.DaskStorage.html#optuna.integration.DaskStorage)
  by specifying a dask ``Client``.
+ More powerful custom search spaces. For example, a variable length list of values can be suggested.
+ It internally uses
  [``optuna.study.Study.optimize``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
  which can be freely configured.
+ It uses ``optuna>=3.1.0``. This is only an advantage until
  [keisuke-umezawa's Pull Request](https://github.com/facebookresearch/hydra/pull/2360) is merged.

Before using consider the following disadvantages:

- This plugin is supposed to be used with the ``BasicLauncher``.
  Different launchers like the [Submitit Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/)
  are not supported.
- The deprecated parameter ``search_space`` from the original has been removed.
- This plugin internally uses [hydra_zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for configuration.
  It requires importing ``optuna`` and ``dask.distributed``.
  Both might be imported even if this plugin is not used and thus, once installed, it can slow down starting hydra.
- This plugin does not seek to improve the architecture of hydra nor does it allow the usage of hyperparameter search
  libraries other than optuna.
- Fewer people use this plugin. Therefore, it might not be as robust as the original. 


## Installation

This plugin requires ``hydra-core>=1.2.0``. Please install it with the following command:
```
pip install hydra-core --upgrade
```
This plugin has not yet been added to PyPI. Install it by cloning this repository and executing
```
pip install PATH-TO-CLONED-REPOSITORY
```
preferably in an activated virtual environment (```pip install -e``` does not work).


## Usage

The ```examples``` package includes the adapted examples provided with the original plugin.
A more complicated deep-learning example can be found here ...

This plugin can mostly be used like the original. However, it has more options.
For more information please take a look at the doc-strings of ``optuna_pruning_sweeper.OptunaPruningSweeper``
and ``custom_search_space.CustomSearchSpace``.
Please set ```hydra/sweeper``` to ```OptunaPruningSweeper``` in your config file.
```yaml
defaults:
  - override hydra/sweeper: OptunaPruningSweeper
```
Alternatively, add the ```hydra/sweeper=OptunaPruningSweeper``` option to your command line.
The default configuration simply consists of the default values of the ```OptunaPruningSweeper```
class in the ```optuna_pruning_sweeper.py``` module.

### Search space configuration

Like the original, this plugin uses hydra's [OverrideGrammer](https://hydra.cc/docs/advanced/override_grammar/extended/).
As a rule of thumb use ``choice`` override instead of ``suggest_categorical`` (i.e. ``x: choice(false, true)``),
``range`` override instead of ``suggest_int`` (i.e. ``x: range(1, 4)``) and ``interval`` override instead of
``suggest_float`` (i.e. ``x: interval(1, 4)``). In case of ``range`` and ``interval`` add the tag "log" for a
logarithmic search space (i.e. ``x: tag(log, range(1, 4))`` and ``x: tag(log, interval(1, 4))``).
Manual values can be specified as tags, i.e. ``x: tag("1", log, interval(1, 4))``.
Unfortunately, hydra throws an error if the tags can be converted to something over than a string.
Therefore, numbers, booleans and ``null`` have to be surrounded by quotes.
If multiple manual values are specified they have to be prefaced by their index, as hydra does not respect the order,
i.e. ``x: tag(0:1, 1:3, log, interval(1, 4))`` (first x=0 is tried then x=3 and later the values are sampled from [1, 4]).
As hydra can't convert ``0:1`` to a number, no quotes are required.

#### Custom Search Space

Override the class
```python
class CustomSearchSpace(ABC):
    def manual_values(self) -> Dict[str, List[Any]]:
        return dict()

    @abstractmethod
    def suggest(self, cfg: DictConfig, trial: Trial) -> Dict[str, Any]:
        pass
```
to get access to the ``trial`` object which can be used to dynamically create the search space.
In contrast to the original Optuna sweeper, the suggested values should be returned with their name in a dictionary.
This allows the suggestion of values which in turn consist of suggested values.
For example, a variable length list of values can be suggested.
This example is already implemented as the ``ListSearchSpace``.
A full example can be found at ``examples/custom_search_space``.


## Implementation Details

### Trial Provider

I want the objective function to be able to access the ``Trial`` object so that the content of
``optuna.integration`` can be reused. Since the hydra launcher interface does not allow this, I simply created
the module ``trial_provider.py`` which only contains the single variable through which the ``Trial``
object can be passed (similar to the singleton design pattern).

### Sweeping

Unlike the original Optuna sweeper, I use
[``optuna.study.Study.optimize``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
method to carry out the sweep.
This should improve the robustness of the code and allow for the usage of further features from optuna
like garbage collection after each trial and more customization options concerning how many trials are run.

### Parallelization

Two methods can be used to parallelize the hyperparameter search.

First, if ``n_jobs>1`` the storage is wrapped in
[``optuna.integration.DaskStorage``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.DaskStorage.html#optuna.integration.DaskStorage)
which is used to start ``num_jobs`` parallel executions of
[``optuna.study.Study.optimize``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
on the specified dask cluster. Libraries like [Dask-Jobqueue](https://jobqueue.dask.org/en/latest/) or
[Dask-MPI](https://mpi.dask.org/en/latest/) can be used to set up a dask cluster over multiple nodes by passing a
callable to the argument ``dask_client`` (see ``examples/sphere_custom_dask_client for an example``).

Another way to parallelize the hyperparameter search is to set up an RDB Backend or
[``optuna.storages.JournalStorage``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html#optuna.storages.JournalStorage)
which each process can access. And start multiple processes from the command line which each carry out the
hyperparameter search, as described [here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html).

### Further Remarks

To improve robustness I mainly rely on
[``optuna.study.Study.optimize``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
and
[``optuna.integration.DaskStorage``](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.DaskStorage.html#optuna.integration.DaskStorage)
. Further, some code has
been adapted from the original [Optuna Sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/) and [keisuke-umezawa's Pull Request](https://github.com/facebookresearch/hydra/pull/2360).


## Issues

+ When the ``DaskStorage`` in combination with an RDB Backend is used some internal exceptions get thrown.
  The hyperparameter search however works as expected.
+ When the ``DaskStorage`` is used, some logging information is not propagated correctly.