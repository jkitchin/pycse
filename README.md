# pycse - Python computations in science and engineering

[![Tests](https://github.com/jkitchin/pycse/actions/workflows/pycse-tests.yaml/badge.svg)](https://github.com/jkitchin/pycse/actions/workflows/pycse-tests.yaml)
[![codecov](https://codecov.io/gh/jkitchin/pycse/branch/master/graph/badge.svg)](https://codecov.io/gh/jkitchin/pycse)
![PyPI Downloads](https://img.shields.io/pypi/dm/pycse.svg)
[![PyPI version](https://badge.fury.io/py/pycse.svg)](https://badge.fury.io/py/pycse)
[![Deploy](https://github.com/jkitchin/pycse/actions/workflows/deploy.yml/badge.svg)](https://github.com/jkitchin/pycse/actions/workflows/deploy.yml)

If you want to cite this project, use this doi:10.5281/zenodo.19111.

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.19111.svg)](http://dx.doi.org/10.5281/zenodo.19111)

```bibtex
@misc{john_kitchin_2015_19111,
  author       = {John R. Kitchin},
  title        = {pycse: First release},
  month        = jun,
  year         = 2015,
  doi          = {10.5281/zenodo.19111},
  url          = {http://dx.doi.org/10.5281/zenodo.19111}
}
```

This git repository hosts my notes on using python in scientific and engineering calculations. The aim is to collect examples that span the types of computation/calculations scientists and engineers typically do to demonstrate the utility of python as a computational platform in engineering education.

## Installation

You may want to install the python library with pycse:

```sh
pip install pycse
```

Feeling brave? You can install the cutting edge from GitHub:

```sh
pip install git+git://github.com/jkitchin/pycse
```

## Docker

You can use a Docker image to run everything here. You have to have Docker installed and working on your system.

See [docker/](./docker/) for the setup used.

### Option 1

I provide a `pycse` command-line utility that is installed with the package. Simply run `pycse` in a shell in the directory you want to start Jupyter lab in. When done, type C-c <return> in the shell to quit, and it should be good.

### Option 2

You can manually pull the image:

```sh
docker pull jkitchin/pycse:latest
```

Then, run the [docker/pycse.sh](./docker/pycse.sh) script. This script mounts the current working directory, and takes care of choosing a random port.

## Documentation

See https://kitchingroup.cheme.cmu.edu/pycse/docs/pycse.html for the Python documentation.
