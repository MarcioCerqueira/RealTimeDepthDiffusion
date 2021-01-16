#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include <iostream>

void GPUAllocateDeviceMemory(int rows, int cols, int levels);
void GPUFreeDeviceMemory(int levels);
void GPULoadWeights(float beta);
void GPUMatrixFreeSolver(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, int level);

#endif