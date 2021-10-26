#!/usr/bin/env python

import os
from setuptools import setup, find_packages

version = "0.0.1"

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst')) as f:
    README = f.read()

REQUIREMENTS = [
    'phe', 'grpc', 'numpy', 'torch', 'scikit-learn'
]

setup(
    name='vfl-knn',
    version=version,
    description='vertical federated learning for KNN',
    long_description=README,
    author='DS3Lab',
    author_email='jiawei.jiang@inf.ethz.ch',
    url='https://github.com/DS3Lab/vfl-knn',
    license="Apache",
    install_requires=REQUIREMENTS,
    keywords=['demo', 'setup.py', 'project'],
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.3',
    ],
    entry_points={
        'console_scripts': ['demo = demo.demo_handler:main']
    },
)