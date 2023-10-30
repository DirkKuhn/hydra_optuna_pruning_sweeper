"""
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch Lightning, and FashionMNIST. We optimize the neural network architecture.
"""
from typing import Literal

from hydra_zen import zen
import torch as th
import pytorch_lightning as pl


def train_nn(
        trainer: pl.Trainer,
        task: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        compile_mode: Literal[False] | str,
        test_monitor: str
) -> float:
    if compile_mode != False:
        task = th.compile(task, mode=compile_mode)

    trainer.fit(task, datamodule=datamodule)
    test_metrics = trainer.test(datamodule=datamodule, ckpt_path='best')
    return test_metrics[0][test_monitor]


zen_run = zen(train_nn, pre_call=lambda cfg: pl.seed_everything(cfg.seed, workers=True))


if __name__ == '__main__':
    import config  # noqa

    # Generate the CLI for run
    zen_run.hydra_main(
        config_name='config',
        config_path=None,
        version_base='1.3'
    )
