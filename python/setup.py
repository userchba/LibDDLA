import os
from setuptools import setup, find_packages

setup(
    name="ddla",
    version="0.1.0",
    description="Python bindings for LibDDLA distributed dense linear algebra",
    author="userchba",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18",
    ],
    extras_require={
        "mpi": ["mpi4py"],
    },
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
