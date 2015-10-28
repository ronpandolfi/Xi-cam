#ifndef WARPIMAGE__H
#define WARPIMAGE__H


#include <Python.h>
#include <numpy/arrayobject.h>

/* setup methdods */
PyMODINIT_FUNC initcWarpImage();
static PyObject * warp_image (PyObject *, PyObject *);

/* utility functions */
float ** pymatrix_to_Carrayptrs (PyArrayObject *);
void free_Carrayptrs(float **);

#endif // WARPIMAGE__H
