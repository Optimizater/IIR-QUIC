#include <Eigen/Dense>
#include <numpy/arrayobject.h>

inline Eigen::MatrixXd numpy_to_eigen(PyArrayObject* arr) {
    return Eigen::Map<Eigen::MatrixXd>(
        (double*)PyArray_DATA(arr),
        PyArray_DIM(arr, 0),
        PyArray_DIM(arr, 1)
    );
}

inline PyObject* eigen_to_numpy(const Eigen::MatrixXd& mat) {
    npy_intp dims[2] = {mat.rows(), mat.cols()};
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    Eigen::Map<Eigen::MatrixXd>(
        (double*)PyArray_DATA((PyArrayObject*)arr),
        mat.rows(),
        mat.cols()
    ) = mat;
    return arr;
}