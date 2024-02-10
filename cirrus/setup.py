from setuptools import setup, find_packages

"""
To import:
import sys
sys.path.append('/Users/simonmyhre/workdir/gitdir/skyline')
from cirrus import Data
"""

setup(
    name="cirrus",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "librosa",
        "numpy",
        "scipy",
    ],
)
