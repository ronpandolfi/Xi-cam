#include "warpimage.h"
#include "remesh.h"
#include <iostream>

/* setup methdods table */
static PyMethodDef cWarpImageMehods [] = {
	{ "warp_image", warp_image, METH_VARARGS },
	{ NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initcWarpImage() {
	(void) Py_InitModule("cWarpImage", cWarpImageMehods);
	import_array();
}

static PyObject * warp_image (PyObject *self, PyObject *args){
	PyObject *arr1, *arr2, *arr3, *arr4, *arr5;
	PyObject *PyImg, *Py_Qp, *Py_Qz, *Pixel, *Center;
    PyArrayObject *PyOut;
	int nrow, ncol;
    int np, nz, outdims[2];
	double alphai, k0, sdd;
    int method;

	/* Parse tuples */
	if (!PyArg_ParseTuple(args, "OOOOOdddI", &arr1, &arr2, &arr3, &arr4, &arr5, 
		&alphai, &k0, &sdd, &method))
		return NULL;

	/* Interpret the input objects as numpy arrays. */
    PyImg = PyArray_FROM_OTF(arr1, NPY_DOUBLE, NPY_IN_ARRAY);
    Py_Qp = PyArray_FROM_OTF(arr2, NPY_DOUBLE, NPY_IN_ARRAY);
    Py_Qz = PyArray_FROM_OTF(arr3, NPY_DOUBLE, NPY_IN_ARRAY);
	Pixel = PyArray_FROM_OTF(arr4, NPY_DOUBLE, NPY_IN_ARRAY);
	Center= PyArray_FROM_OTF(arr5, NPY_DOUBLE, NPY_IN_ARRAY);

	if (PyImg == NULL || Py_Qp == NULL || Py_Qz == NULL ||
		Pixel == NULL || Center == NULL){
		Py_XDECREF(PyImg);
		Py_XDECREF(Py_Qp);
		Py_XDECREF(Py_Qz);
		Py_XDECREF(Pixel);
		Py_XDECREF(Center);
		return NULL;
	}

	/* matrix dimensions */
	nrow = (int) PyArray_DIM(PyImg, 0);
	ncol = (int) PyArray_DIM(PyImg, 1);

    /* output dimensions */
    np = outdims[0] = (int) PyArray_DIM(Py_Qp, 0);
    nz = outdims[1] = (int) PyArray_DIM(Py_Qp, 1);
     
	/* contruct output array */
	PyOut = (PyArrayObject *) PyArray_FromDims (2, outdims, NPY_DOUBLE);

	/* map PyArrays to c-arrays */
	double * img = (double *) PyArray_DATA(PyImg);
	double *qpar = (double *) PyArray_DATA(Py_Qp);
	double *qvrt = (double *) PyArray_DATA(Py_Qz);
	double *pixel= (double *) PyArray_DATA(Pixel);
	double *cen  = (double *) PyArray_DATA(Center);
	double * out = (double *) PyArray_DATA(PyOut);

    // if(knn_interp (nrow*ncol, img, crd1, crd2, nx, ny, out, xout, yout, method))
    //    return NULL;
	if(!(remap(nrow, ncol, img, np, nz, qpar, qvrt, pixel, cen, alphai, k0, sdd, method, out)))
		return NULL;

	///*** C calls end ***/
	//
	Py_XDECREF(PyImg);
	Py_XDECREF(Py_Qp);
	Py_XDECREF(Py_Qz);
	Py_XDECREF(Pixel);
	Py_XDECREF(Center);
	return PyArray_Return (PyOut);
}
