"""
@Author: Rossi
2016-02-01
"""
from os.path import dirname, join
from setuptools import setup, find_packages


with open("requrements.txt") as fi:
    dependencies = [line.strip() for line in fi]


setup(
    name="ner",
    version="0.1",
    description="named entity recognition tool",
    author="Rossi",
    packages=find_packages(exclude=("test", "test.*")),
    install_requires=dependencies,
)
