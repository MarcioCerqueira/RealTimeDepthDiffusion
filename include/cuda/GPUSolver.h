#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include <iostream>

void GPUAllocateDeviceMemory(int rows, int cols, int levels);
void GPUFreeDeviceMemory(int levels);
void GPULoadWeights(float beta);
void GPUMatrixFreeSolver(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, std::string solverName, std::string method, 
	int level, bool isDebugEnabled);
void CUSPMatrixFreeSolver(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols, 
	float beta, int maxIterations, float tolerance, std::string solverName, bool isDebugEnabled);
void CUSPPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols, float beta,
	int maxIterations, float tolerance, bool isDebugEnabled);
void ParalutionPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols,
	float beta, int maxIterations, float tolerance);
void ViennaCLPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols,
	float beta, int maxIterations, float tolerance);

#endif