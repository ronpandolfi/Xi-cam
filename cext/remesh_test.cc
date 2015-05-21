#include <iostream>
#include <cmath>
#include "remesh.h"

const int nrow=1024;
const int ncol=1024;
const int Np = 4;
const int Nd = 2;

int main ()
{
    int i, j, ipos, jpos;
    float **xcrd, **ycrd;
    float dist;
    float pts[Np][Nd] = {127, 179, 233, 239, 1421, 479, 557, 1613};


    // allocate memory
    xcrd = new float *[nrow];
    for (i=0; i<nrow; i++)
        xcrd[i] = new float[ncol];   

    ycrd = new float *[nrow];
    for (i=0; i<nrow; i++)
        ycrd[i] = new float[ncol];   

    for (i=0; i<nrow; i++)
        for (j=0; j<ncol; j++) {
            xcrd[i][j] = j;
            ycrd[i][j] = i;
        }

    for (i=0; i < Np; i++) {
        nearest_neighbor (nrow, ncol, xcrd, ycrd, pts[i], ipos, jpos, dist);
        std::cout << "Point : " << pts[i][0] << ", " << pts[i][1] << std::endl;
        std::cout << "Nearest Neigh :" << xcrd[ipos][jpos] << ", " << ycrd[ipos][jpos] << std::endl;
        std::cout << "Dist = " << std::sqrt(dist) << std::endl;
        std::cout << "-----------------" << std::endl;
    }
}
