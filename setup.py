"""
DEPRECATED: This file is kept for backward compatibility.
Please use pyproject.toml for all project configuration.
"""

import warnings

warnings.warn(
    "setup.py is deprecated. Please use pyproject.toml for project configuration.",
    DeprecationWarning,
    stacklevel=2,
)

from setuptools import setup

if __name__ == "__main__":
    setup()
