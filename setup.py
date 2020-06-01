from pathlib import Path
from typing import List
from setuptools import setup


def parse_requirements() -> List[str]:
    path = Path() / 'requirements.txt'
    with open(str(path), 'r') as file:
        data = file.read()
    lines = data.split('\n')
    return [line.strip() for line in lines if line.strip()]


setup(name='spark_lp',
      version='0.0.1',
      packages=['spark_lp'],
      install_requires=parse_requirements()
      )
