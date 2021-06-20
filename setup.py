"""
@Author: Rossi
2016-02-01
"""
import os.path
from setuptools import setup, find_packages


setup(
    name="bert_ner",
    version="0.1",
    description="named entity recognition tool",
    author="Rossi",
    packages=find_packages(exclude=("test", "test.*"))
)
