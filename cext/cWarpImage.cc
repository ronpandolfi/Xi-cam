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

float * pymatrix_to_C1darrayptrs (PyArrayObject *arrayin) {
    return (float *) arrayin->data;
}

float ** pymatrix_to_C2darrayptrs (PyArrayObject *arrayin)  {
    float **c, *a;
    int i,n,m;
 
    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c= new float *[n];
    a=(float *) arrayin->data;  /* pointer to arrayin data */
    for ( i=0; i<n; i++)  {
         c[i]=a+i*m;  }
     return c;
 }

/* ==== Free vec of pointers ============== */ 
void free_Carrayptrs(float **v)  {
    delete [] v;
}

 
static PyObject * warp_image (PyObject *self, PyObject *args)
{
	PyArrayObject *PyImg, *PyCrd1, *PyCrd2;
    PyArrayObject *PyXout, *PyYout, *PyOut;
	float *img, *crd1, *crd2; 
    float *xout, *yout, *out;
	int nrow, ncol, dims[2];
    int nx, ny, outdims[2];
    int method;

	/* Parse tuples */
	if (!PyArg_ParseTuple(args, "OOOOOI", &PyImg, &PyCrd1, &PyCrd2, &PyXout, &PyYout, &method))
		return NULL;
	if (NULL == PyImg) return NULL;
	if (NULL == PyCrd1) return NULL;
	if (NULL == PyCrd2) return NULL;
	if (NULL == PyXout) return NULL;
	if (NULL == PyYout) return NULL;

	/* matrix dimensions */
	nrow = dims[0] = PyImg->dimensions[0];
	ncol = dims[1] = PyImg->dimensions[1];

    /* output dimensions */
    ny = outdims[0] = PyXout->dimensions[0];
    nx = outdims[1] = PyXout->dimensions[1];
     
	/* contruct output array */
	PyOut = (PyArrayObject *) PyArray_FromDims (2, outdims, NPY_FLOAT);

	/* change Numpy arrays to 2-D c arrays */
	img = pymatrix_to_C1darrayptrs (PyImg);
	out = pymatrix_to_C1darrayptrs (PyOut);
	crd1 = pymatrix_to_C1darrayptrs (PyCrd1);
	crd2 = pymatrix_to_C1darrayptrs (PyCrd2);
	xout = pymatrix_to_C1darrayptrs (PyXout);
	yout = pymatrix_to_C1darrayptrs (PyYout);

	/*** C calls begin ***/
    //    if (remesh (nrow, ncol, img, crd1, crd2, nx, ny, out, xout, yout))
    //    return NULL;

    
    if (knn_interp (nrow*ncol, img, crd1, crd2, nx, ny, out, xout, yout, method))
        return NULL;
	/*** C calls end ***/
	
	return PyArray_Return (PyOut);
}
