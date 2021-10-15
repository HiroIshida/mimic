import sys
from setuptools import setup, find_packages

setup_requires = []

install_requires = [
        "numpy",
        "torch",
        "torchvision",
        "tqdm",
        ]

# for running demo
extras_require = {
        'test': ["pybullet", "tinyfk", "pybullet", "moviepy", "matplotlib"]
        }

setup(
    name='mimic',
    version='0.0.1',
    description='',
    license=license,
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(exclude=('tests', 'docs'))
)
