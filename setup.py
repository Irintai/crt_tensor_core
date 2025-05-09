#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

# Version information
VERSION = '0.4.0'  # Version
DESCRIPTION = 'Tensor operations based on Cosmological Recursion Theory principles'
LONG_DESCRIPTION = '''
# CRT Tensor Core

A Python/Cython library implementing tensor operations based on Cosmological Recursion Theory (CRT) principles.

This library provides efficient tensor operations optimized for CRT calculations, making advanced cosmological 
modeling more accessible and computationally efficient.

For more information, see the [documentation](https://crt-tensor-core.readthedocs.io/). ** Coming soon **
'''

# Package meta-data
AUTHOR = 'Andrew \'Irintai\' Orth'
AUTHOR_EMAIL = 'drewski871@gmail.com'  # Removed typo 'L' at the end
URL = 'https://github.com/Irintai/crt_tensor_core'
LICENSE = 'MIT'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering :: Astronomy',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Mathematics',
]

# Define Cython extensions
extensions = [
    Extension(
        "crt_tensor_core.kernels",
        ["kernels.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Requirements
REQUIRED = [
    'numpy>=1.19.0',
    'scipy>=1.7.0',
    'matplotlib>=3.3.0',
]

EXTRAS = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.900',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'nbsphinx>=0.8.0',
        'ipython>=7.0.0',
    ],
    'notebooks': [
        'jupyter>=1.0.0',
        'pandas>=1.3.0',
    ],
}

# The main setup function
setup(
    name="crt_tensor_core",
    version="0.4.0",
    author="Andrew 'Irintai' Orth",
    author_email="drewski871@gmail.com",
    description="A tensor library for Cosmological Recursion Theory",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/irintai/crt_tensor_core",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    python_requires=">=3.8",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    ext_modules=cythonize(extensions, compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
    }),
    include_dirs=[np.get_include()],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "crt-benchmark=crt_tensor_core.tools.benchmark:main",
        ],
    },
)