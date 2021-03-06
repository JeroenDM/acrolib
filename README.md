[![Build Status](https://travis-ci.org/JeroenDM/acrolib.svg?branch=master)](https://travis-ci.org/JeroenDM/acrolib) [![codecov](https://codecov.io/gh/JeroenDM/acrolib/branch/master/graph/badge.svg)](https://codecov.io/gh/JeroenDM/acrolib)

# Installation

First install cython, wich `setup.py` needs to build the package.
```bash
pip install cython
```

In some cases you also have to install additional dependencies.
```bash
sudo apt install python3-dev
pip install wheel
```

## Using pip
Then install the package.
```bash
pip install acrolib
```

## From source
```bash
git clone https://github.com/JeroenDM/acrolib.git
cd acrolib
python setup.py build
python setup.py install
```
If you want to edit the package and test the changes, you can replace the last line with:
```bash
python setup.py develop
```

# Acrolib

General utilities for writing motion planning algorithms at [ACRO](https://iiw.kuleuven.be/onderzoek/acro).
This library is aimed at miscellaneous functions and classes that cannot be grouped in a larger package.

## Dynamic Programming

Solve a specific type of Deterministic Markov Decision Process.
It uses a value function that must be minimized instead of maximized.
It assumes a sequential linear graph structure.

## Quaternion

Extension to the [pyquaternion](http://kieranwynn.github.io/pyquaternion/) package.

## Sampling

A sampler class to generate uniform random or deterministic samples.
Deterministic samples are generated using a [Halton Sequence](https://en.wikipedia.org/wiki/Halton_sequence).
