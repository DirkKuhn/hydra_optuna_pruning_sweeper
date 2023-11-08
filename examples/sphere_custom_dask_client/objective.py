import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def sphere(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    return x**2 + y**2


def setup_dask(n_workers: int):
    from dask.distributed import Client
    return Client(n_workers=n_workers)


if __name__ == "__main__":
    sphere()
