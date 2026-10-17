from pathlib import Path

from setuptools import find_packages, setup


requirements = [
    line.strip()
    for line in (Path(__file__).parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="aegis_orbit",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
)