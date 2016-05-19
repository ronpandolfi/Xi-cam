#include "remesh.h"
#include <iostream>
#include <cmath>

bool remap(int nrow, int ncol, double *img, int nqp, int nqz, 
		double *qp, double *qz, double *pixel, double *cen,
		double alphai, double k0, double sdd, int method, 
		double *out){

    for (int i = 0; i < 2; i++)
        cen[i] /= pixel[i];
	double sin_ai = std::sin(alphai);
	double cos_ai = std::cos(alphai);
	int ir, ic;
#pragma omp parallel for private(ir, ic)
	for (int i = 0; i < nqp * nqz; i++){
		double sina = (qz[i] / k0) - sin_ai;
		double cosa = sincos(sina);
		double tana = sina / cosa;

		double t0 = qp[i] / k0;
		double t1 = cosa * cosa + cos_ai * cos_ai - t0 * t0;
		double t2 = 2. * cosa * cos_ai;
		if (t1 > t2){
			out[i] = 0.;
			continue;
		}
		double cost = t1 / t2;
		double tant = sgn(qp[i]) * sincos(cost) / cost;
		
		double row = tana * sdd / cost / pixel[1] + cen[1];
		double col = tant * sdd / pixel[0] + cen[0];
		switch(method){
			case NEAREST_NEIGHBOR:
				ir = (int) (row + 0.5);
				ic = (int) (col + 0.5);
				if (ic >= 0 && ic < ncol && ir >= 0 && ir < nrow )
					out[i] = img[ir * ncol + ic];
				else
					out[i] = 0.;
				break;


			case BILINEAR:
				break;

			case CUBIC_SPLINE:
				break;
		}
	}
	return true;
}
