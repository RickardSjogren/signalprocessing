#!/usr/bin/env python

from distutils.core import setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name='Signal-processing utils',
    version='0.1.1',
    description='Signal processing utilities for Airgard-project',
    long_description=long_description,
    author='Rickard Sjögren',
    author_email='rickard.sjogren@umu.se',
    packages=['signalprocessing_utils'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'pywavelets',
        'scipy'
    ]
)