#ifndef REMESH__H
#define REMESH__H

#ifdef __cplusplus
extern "C" {
#endif

enum method_t { NEAREST_NEIGHBOR, GAUSSIAN };
int nearest_neighbor (int , int , float **, float **, float *, int &, int &, float &);
int remesh (int, int, float **, float **, float **, int, int, float **, float **, float **);
int knn_interp (int, float *, float *, float *, int, int, float *, float *, float *, int);

#ifdef __cplusplus
}
#endif

#endif // REMESH__H
