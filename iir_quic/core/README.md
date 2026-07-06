# Introduction

This module is modified based on the [QUIC code](https://github.com/osdf/pyquic) and provides the code for solving sub-problems in DIIR-QUIC.

The code is this part has only been tested on Ubuntu 22.04

# How to build

To build the module, you need to have `lapack` installed.

For Ubuntu, you can install it using the following command:

```bash
sudo apt-get install libblas-dev liblapack-dev
```

Then, you can build the module by running the following command in the terminal:

```bash
# cd core
python setup.py build_ext --inplace
```
