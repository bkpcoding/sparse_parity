from setuptools import setup, find_packages

setup(
    name="sparse_parity",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        "numpy",
        "torch",
        "wandb",
        "hydra-core",
        "tqdm"
    ]
)