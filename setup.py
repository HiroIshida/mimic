import sys
from setuptools import setup, find_packages

setup_requires = []

install_requires = [
        "numpy",
        "torch",
        "tqdm",
        ]

# for running demo
extra_require = [
        "pybullet",
        "tinyfk",
        "pybullet",
        "moviepy"
        ]

setup(
    name='mimic',
    version='0.0.1',
    description='',
    license=license,
    install_requires=install_requires,
    extra_require=extra_require,
    packages=find_packages(exclude=('tests', 'docs'))
)
