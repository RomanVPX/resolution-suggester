from setuptools import setup, find_packages

setup(
    name="resolution-suggester",
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
            "resolution-suggester=resolution-suggester.main:main",
        ],
    },
)