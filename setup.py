from setuptools import setup, find_packages

setup(
    name="resolution_suggester",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "resolution_suggester=resolution_suggester.main:main",
            "res-suggest=resolution_suggester.main:main",
        ],
    },
)
