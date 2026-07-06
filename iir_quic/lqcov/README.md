# Introduction

This module provides implementation of the Lq-optimization approach for sparse inverse covariance selection,
based on the paper:
On Lq Optimization and Sparse Inverse Covariance Selection (Goran Marjanovic et al., 2014)

# How to build

To build the module, you need to have `eigen` installed.

1. For Ubuntu, you can install it using the following command:

```bash
sudo apt-get install libeigen3-dev
```

2. For Windows, you can download the Eigen library from its [official website](https://eigen.tuxfamily.org/dox/GettingStarted.html) and follow the instructions to set it up.

   In short, you need to download the compressed package, unzip it locally and modify the `eigen_path` in [setup.py](setup.py) to your folder path where you unzipped the Eigen library.

Then, you can build the module by running the following command in the terminal:

```bash
# cd lqcov
python setup.py build_ext --inplace
```

# How to use

This module provides two versions of the Lq-optimization approach:

1. pure python version (lqcov) for easy understanding and debugging
2. C++ version (lqcov_cython) for better performance

You can use either version by importing the corresponding module in your Python code.

```python
# Example usage
import diir_quic.lqcov.lqcov as lqcov_m1
import diir_quic.lqcov.lqcov_cython as lqcov_m2
```

Each version provides the same interface:

1. `lq_cov`: lq COV algorithm for sparse inverse covariance selection. (Original version from the paper)
2. `lq_cov2`: lq COV algorithm for sparse inverse covariance selection, using stationary condition.
3. `lq_cov3`: lq COV algorithm for sparse inverse covariance selection, using stationary condition and time limitation.
