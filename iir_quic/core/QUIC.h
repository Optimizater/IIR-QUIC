#ifndef _QUIC_IRL_H
#define _QUIC_IRL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// void QUIC(char mode, uint32_t& p, const double* S, double* Lambda0,
// 	  uint32_t& pathLen, const double* path, double& tol,
// 	  int32_t& msg, uint32_t& maxIter,
// 	  double* X, double* W, double* opt, double* cputime,
// 	  uint32_t* iter, double* dGap);
void QUIC(char mode, uint32_t &p, const double *S, double *Lambda0,
	uint32_t &pathLen, const double *path, double &tol,
	int32_t &msg, uint32_t &maxIter,
	double *X, double *W, double *opt, double *cputime,
	uint32_t *iter, double *dGap, double *info_list);

#ifdef __cplusplus
}
#endif

#endif
