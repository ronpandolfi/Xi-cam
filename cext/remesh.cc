#include "remesh.h"
#include "kdtree2.hpp"
#include <vector>
#include <boost/multi_array.hpp>
#include <cstdlib>
#include <iostream>
#include <cmath>



int nearest_neighbor (int nrow, int ncol, float **qr, float **qz,
    float *pt, int &ipos, int &jpos, float & dist)
{
    int ibeg = 0, iend=nrow-1, jbeg=0, jend=ncol-1;
    int icen = (ibeg + iend)/2;
    int jcen = (jbeg + jend)/2;
    int i, j;
    float dr, dz;

    while ((iend-ibeg > 100) && (jend-jbeg > 100)) {
        dr = pt[0] - qr[icen][jcen];
        dz = pt[1] - qz[icen][jcen];

        if ( dz < 0 ) {
            iend = icen;
            if ( dr < 0 )
                jend = jcen;
            else
                jbeg = jcen;
        } else {
            ibeg = icen;
            if ( dr < 0 )
                jend = jcen;
            else
                jbeg = jcen;
        } 

        icen = (ibeg + iend)/2;
        jcen = (jbeg + jend)/2;
    }

    // check if bounding box has y-part
    unsigned found = 0;
    //std::cerr << "found = " << found << std::endl;
    //std::cerr << "ibeg = " << ibeg << ", iend " << iend << std::endl;
    //std::cerr << "jbeg = " << jbeg << ", jend " << jend << std::endl;
    for (j = jbeg; j <= jend; j++)
        if ((qz[ibeg][j] <= pt[1]) && (qz[iend][j] >=  pt[1])) {
            found |=  2;
            break;
        }

    // check if bounding box has x-part
    for (i = ibeg; i <= iend; i++)
        if ((qr[i][jbeg] <= pt[0]) && (qr[i][jend] >= pt[0])) {
            found |= 1;
            break;
        }

    float dmin = 1.0E+10;
    if ( found == 0x3 ){
        for (i=ibeg; i<=iend; i++)
            for (j=jbeg; j<=jend; j++) {
                dr = pt[0] - qr[i][j];
                dz = pt[1] - qz[i][j];
                if (dr*dr+dz*dz < dmin) {
                    dmin = dr*dr + dz*dz;
                    icen = i;
                    jcen = j;
                }
            }
    } else if (found == 0x2) {
        if ( std::abs(pt[0]-qr[icen][0]) < std::abs(pt[0]-qr[icen][ncol-1]))
            jcen = 0;
        else
            jcen = ncol-1;
        for (i = ibeg; i <= iend; i++) {
            dr = pt[0] - qr[i][jcen];
            dz = pt[1] - qz[i][jcen];
            if (dr*dr + dz*dz < dmin) {
                dmin = dr*dr + dz*dz;
                icen = i;
                
            }
        }
    } else if (found == 0x1) {
        if (std::abs(pt[1]-qz[0][jcen]) < std::abs(pt[1]-qz[nrow-1][jcen]))
            icen = 0;
        else
            icen = nrow-1;
        for (j = jbeg; j < jend; j++) {
            dr = pt[0] - qr[icen][j];
            dz = pt[1] - qz[icen][j];
            if (dr*dr + dz*dz < dmin) {
                dmin = dr*dr + dz*dz;
                jcen = j;
            }
        } 
    } else return 1;

    ipos = icen;
    jpos = jcen;
    dist = dmin;
    return 0;
}

int remesh (int nrow, int ncol, float **img, float **crd1, float **crd2,
    int nx, int ny, float **output, float **xout, float **yout) {

	int i, j;
	int ipos, jpos;
	float pt[2];
	float dx,dy,rad,dist;

	dx = xout[1]-xout[0]; 
	dy = yout[nx] - yout[0];
	rad = dx*dx + dy*dy;
	dist = 1.0E+10;

    

	for (i = 0; i < ny; i++) {
		for (j = 0; j < nx; j++) {
			pt[0] = xout[i][j];
			pt[1] = yout[i][j];
            
			if (nearest_neighbor (nrow, ncol, crd1, crd2, pt, ipos, jpos, dist)) {
				std::cerr <<"Nearest-Neighbor search failed." << std::endl;
				exit(1);
			}
            //std::cerr <<" Pt = [ " << pt[0] << ", " << pt[1] << " ]\n";
            //std::cerr <<" Neigh = [ " << crd1[ipos][jpos] << ", " << crd2[ipos][jpos] << " ]\n";
            //std::cerr << "dist = " << dist << ", rad = " << rad << std::endl;
			if ( dist < rad ) {
				output[i][j] = img[ipos][jpos];
            }
            else
                output[i][j] = 0;
		}
    }
    return 0;
}


int knn_interp (int size, float *img, float *crd1, float *crd2,
    int nx, int ny, float *output, float *xout, float *yout, int method) {

	int i, ipos;
	float dx, dy, dist;
    const int size2 = nx*ny;

    boost::multi_array<float, 2> points(boost::extents[size][2]);
	dx = xout[1] - xout[0];
	dy = yout[nx] - yout[0];
	float rad2 = dx*dx + dy*dy;
    float rad = std::sqrt(rad2);
    float radinv = 0.5/rad2;

    for (i=0; i < size; i++) {
        points[i][0] = crd1[i];
        points[i][1] = crd2[i]; 
    }
    kdtree2 * tree = new kdtree2(points, true);
    tree->sort_results = true;

#pragma omp parallel for private (ipos)
	for (i = 0; i < size2; i++) {
        kdtree2_result_vector neighs;
        kdtree2_result_vector::iterator itr;
        std::vector <float> pt(2);
		pt[0] = xout[i];
		pt[1] = yout[i];

        switch (method) {
            case NEAREST_NEIGHBOR:
                tree->n_nearest (pt, 1, neighs); 
                //std::cerr << pt[0] << ", " << pt[1] << ", " << crd1[ipos] << ", " << crd2[ipos] << std::endl;
                ipos = neighs[0].idx;
                dist = (pt[0]-crd1[ipos])*(pt[0]-crd1[ipos]) + (pt[1]-crd2[ipos])*(pt[1]-crd2[ipos]);
                if ( dist < rad2 ) {
				    output[i] = img[ipos];
                } 
                else 
                    output[i] = 0;
                break;
    
            case GAUSSIAN:
                tree->n_nearest (pt, 20, neighs); 
                float numer = 0, denom = 0;
                for (itr = neighs.begin(); itr != neighs.end(); itr++) {
                    if (itr->dis < rad2 ) {
                        float t = radinv * itr->dis;
                        float e = std::exp (-t);
                        numer += img[itr->idx] * e;
                        denom += e;
                    }
                }
                if ( denom > 0 )    
                    output[i] = numer/denom;
                else 
                    output[i] = 0;
                break;
        }
    }
    return 0;
}
