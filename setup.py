"""
@Author: Rossi
2016-02-01
"""
import os.path
from setuptools import setup, find_packages

dependency_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

with open(dependency_file) as fi:
    dependencies = [line.strip() for line in fi]


setup(
    name="bert_ner",
    version="0.1",
    description="named entity recognition tool",
    author="Rossi",
    packages=find_packages(exclude=("test", "test.*")),
    install_requires=dependencies,
)
