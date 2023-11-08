# type: ignore
from setuptools import find_namespace_packages, setup

with open("README.md") as fh:
    setup(
        name="hydra-optuna-pruning-sweeper",
        version="0.0.1",
        author="Dirk Kuhn",
        description="Hydra optuna sweeper plugin which supports pruning",
        long_description=fh.read(),
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/hydra/",
        packages=find_namespace_packages(include=["hydra_plugins.*"]),
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Operating System :: OS Independent"
        ],
        install_requires=[
            "hydra-core>=1.2.0",
            "hydra_zen>=0.11.0",
            "optuna>=3.1.0",
            "distributed>=2023.10.1"
        ],
        include_package_data=True
    )
