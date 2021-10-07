import sys
from setuptools import setup, find_packages

setup_requires = []

install_requires = [
        "numpy",
        "torch",
        "tqdm",
        ]

setup(
    name='mimic',
    version='0.0.1',
    description='',
    license=license,
    install_requires=install_requires,
    packages=find_packages(exclude=('tests', 'docs'))
)
