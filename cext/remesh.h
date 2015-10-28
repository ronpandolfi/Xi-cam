#ifndef REMESH__H
#define REMESH__H

#include <cmath>

const int NEAREST_NEIGHBOR = 0;
const int BILINEAR = 1;
const int CUBIC_SPLINE = 2;

inline double sincos(double x){ 
	if (x < 1.)
		return std::sqrt(1. - x * x);
	else
		return 0;
}

inline double sgn(double x) { 
	if (x < 0) return -1;
	else return 1;
}

int knn_interp (int, float *, float *, float *, int, int, float *, float *, float *, int);
bool remap(
		int NROW, 
		int NCOL, 
		double *IMAGE, 
		int NQ_PAR,
		int NQ_VRT,
		double *QP,
		double *QV,
		double *PIXEL_SIZE,
		double *CENTER,
		double ALPHA_INCENDENT,
		double K0,
		double SDD,
		int METHOD,
		double *OUT_IMAGE
		);

#endif // REMESH__H
