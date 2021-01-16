#define TILE_WIDTH 16

#include "GPUSolver.h"
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>

//Global variables shared by the distinct solvers
float **devicePreviousImage;
float **deviceNextImage;
float **deviceDepthImage;
float **deviceError;
int2 **deviceIndexToWeight;
int maxLevel;
__device__ __constant__ float deviceWeights[257];

void GPUCheckError(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

int divUp(int a, int b) { 
    return (a + b - 1)/b;
}

void GPUAllocateDeviceMemory(int rows, int cols, int levels) {
	
	devicePreviousImage = (float**)malloc(sizeof(float*) * levels);
	deviceNextImage = (float**)malloc(sizeof(float*) * levels);
	deviceDepthImage = (float**)malloc(sizeof(float*) * levels);
	deviceError = (float**)malloc(sizeof(float*) * levels);
	deviceIndexToWeight = (int2**)malloc(sizeof(int2*) * levels);

	for(int level = 0; level < levels; level++) {
		int rowsPerLevel = rows / powf(2, level);
		int colsPerLevel = cols / powf(2, level);
		cudaMalloc((void**)&devicePreviousImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceNextImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceDepthImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceError[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceIndexToWeight[level], sizeof(int2) * rowsPerLevel * colsPerLevel);
	}

	maxLevel = levels - 1;
	GPUCheckError("GPUAllocateDeviceMemory");
	
}

void GPUFreeDeviceMemory(int levels) {

	for(int level = 0; level < levels; level++) {
		cudaFree(devicePreviousImage[level]);
		cudaFree(deviceNextImage[level]);
		cudaFree(deviceDepthImage[level]);
		cudaFree(deviceError[level]);
		cudaFree(deviceIndexToWeight[level]);
	}

#ifdef PARALUTION
	paralution::stop_paralution();
#endif
	GPUCheckError("GPUFreeDeviceMemory");

}

__device__ float solveDiffusion(int left, int right, int up, int down, float sharedImage[][TILE_WIDTH + 2], int tidx, int tidy) {

	float weight = 0;
	float sum = 0;
	float count = 0;
	
	if (left != 256) {
		weight = deviceWeights[left];
		sum += weight * sharedImage[tidy][tidx - 1];
		count += weight;
	}

	if (right != 256) {
		weight = deviceWeights[right];
		sum += weight * sharedImage[tidy][tidx + 1];
		count += weight;
	}
	
	if (up != 256) {
		weight = deviceWeights[up];
		sum += weight * sharedImage[tidy - 1][tidx];
		count += weight;
	}
	
	if (down != 256) {
		weight = deviceWeights[down];
		sum += weight * sharedImage[tidy + 1][tidx];
		count += weight;
	}
	
	if (count == 0) return 0.0;
	else return min(max(sum / count, 0.0), 255.0);

}

__global__ void copyFromPitchedData(float *output, float *input, size_t inputPitch, int rows, int cols) 
{
	
	const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	if(x >= cols || y >= rows) return;

	float *inputRow = (float*)((char*)input + y * inputPitch);
	int pixel = __mul24(y, cols) + x;
	output[pixel] = inputRow[x];

}

__global__ void copyToPitchedData(float *output, float *input, size_t outputPitch, int rows, int cols) 
{
	
	const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	if(x >= cols || y >= rows) return;

	float *outputRow = (float*)((char*)output + y * outputPitch);
	int pixel = __mul24(y, cols) + x;
	outputRow[x] = input[pixel];

}

__global__ void loadIndexToWeight(unsigned char *grayImage, float *depthImage, int2 *indexToWeight, size_t grayPitch, size_t depthPitch, int level, 
	int maxLevel, int rows, int cols)
{

	const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	if(x >= cols || y >= rows) return;

	__shared__ unsigned char sharedGrayImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	__shared__ unsigned char sharedDepthImage[TILE_WIDTH + 2][TILE_WIDTH + 2];

	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;

	unsigned char* grayImageRow = grayImage + y * grayPitch;
	float* depthImageRow = (float*)((char*)depthImage + y * depthPitch);
	sharedGrayImage[tidy][tidx] = grayImageRow[x];
	
	if(tidx == 1 && x - 1 >= 0) sharedGrayImage[tidy][0] = grayImageRow[x - 1];
	if(tidx == TILE_WIDTH && x + 1 < cols) sharedGrayImage[tidy][TILE_WIDTH + 1] = grayImageRow[x + 1];
	if (tidy == 1 && y - 1 >= 0) {
		unsigned char* grayImageRowMinus = grayImage + (y - 1) * grayPitch;
		sharedGrayImage[0][tidx] = grayImageRowMinus[x];
	}
	if (tidy == TILE_WIDTH && y + 1 < rows) {
		unsigned char* grayImageRowPlus = grayImage + (y + 1) * grayPitch;
		sharedGrayImage[TILE_WIDTH + 1][tidx] = grayImageRowPlus[x];
	}
	
	if(level != maxLevel) {
		
		sharedDepthImage[tidy][tidx] = depthImageRow[x];
		if(tidx == 1 && x - 1 >= 0) sharedDepthImage[tidy][0] = depthImageRow[x - 1];
		if(tidx == TILE_WIDTH && x + 1 < cols) sharedDepthImage[tidy][TILE_WIDTH + 1] = depthImageRow[x + 1];
		if (tidy == 1 && y - 1 >= 0) {
			float* depthImageRowMinus = (float*)((char*)depthImage + (y - 1) * depthPitch);
			sharedDepthImage[0][tidx] = depthImageRowMinus[x];
		}
		if (tidy == TILE_WIDTH && y + 1 < rows) {
			float* depthImageRowPlus = (float*)((char*)depthImage + (y + 1) * depthPitch);
			sharedDepthImage[TILE_WIDTH + 1][tidx] = depthImageRowPlus[x];
		}
	}
	
	__syncthreads();
	
	int left = 256;
	int right = 256;
	int up = 256;
	int down = 256;
	
	if(level == maxLevel) {
		
		unsigned char grayIntensity = sharedGrayImage[tidy][tidx];
		if(x - 1 >= 0) left = __sad(grayIntensity, sharedGrayImage[tidy][tidx - 1], 0);
		if(x + 1 < cols) right = __sad(grayIntensity, sharedGrayImage[tidy][tidx + 1], 0);
		if(y - 1 >= 0) up = __sad(grayIntensity, sharedGrayImage[tidy - 1][tidx], 0);
		if(y + 1 < rows) down = __sad(grayIntensity, sharedGrayImage[tidy + 1][tidx], 0);
	
	} else {
		
		unsigned char grayIntensity = sharedGrayImage[tidy][tidx];
		unsigned char depthIntensity = sharedDepthImage[tidy][tidx];

		int threshold = 4;
		if(level == 0) threshold = 0;
		if(x - 1 >= 0) {
			if(__sad(depthIntensity, sharedDepthImage[tidy][tidx - 1], 0) > threshold) left = __sad(grayIntensity, sharedGrayImage[tidy][tidx - 1], 0);
			else left = 0;
		}
		if(x + 1 < cols) {
			if(__sad(depthIntensity, sharedDepthImage[tidy][tidx + 1], 0) > threshold) right = __sad(grayIntensity, sharedGrayImage[tidy][tidx + 1], 0);
			else right = 0;
		}
		if(y - 1 >= 0) {
			if(__sad(depthIntensity, sharedDepthImage[tidy - 1][tidx], 0) > threshold) up = __sad(grayIntensity, sharedGrayImage[tidy - 1][tidx], 0);
			else up = 0;
		}
		if(y + 1 < rows) {
			if(__sad(depthIntensity, sharedDepthImage[tidy + 1][tidx], 0) > threshold) down = __sad(grayIntensity, sharedGrayImage[tidy + 1][tidx], 0);
			else down = 0;
		}
		
	}
	
	indexToWeight[y * cols + x] = make_int2(__mul24(left, 1000) + right, __mul24(up, 1000) + down);

}

__global__ void matrixFreeSolver(float *input, int2 *indexToWeight, unsigned char *scribbleImage, float *error, size_t inputPitch, size_t scribblePitch, int rows, 
	int cols, float *jacobiOutput = 0, float *chebyshevPreviousImage = 0, float omega = 0, float gamma = 0) 
{

	const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	
	if (x >= cols || y >= rows) return;

	int pixel = __mul24(y, cols) + x;
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ float sharedImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	sharedImage[tidy][tidx] = input[pixel];
	if (tidx == 1 && x - 1 >= 0) sharedImage[tidy][0] = input[pixel - 1];
	if (tidx == TILE_WIDTH && x + 1 < cols) sharedImage[tidy][TILE_WIDTH + 1] = input[pixel + 1];
	if (tidy == 1 && y - 1 >= 0) sharedImage[0][tidx] = input[(y - 1) * cols + x];
	if (tidy == TILE_WIDTH && y + 1 < rows) sharedImage[TILE_WIDTH + 1][tidx] = input[(y + 1) * cols + x];
	__syncthreads();
	
	unsigned char *scribbleImageRow = scribbleImage + y * scribblePitch;
	if(scribbleImageRow[x] == 255) return;
	
	int2 index = indexToWeight[pixel];
	int left = index.x / 1000;
	int right = index.x % 1000;
	int up = index.y / 1000;
	int down = index.y % 1000;
	float result = solveDiffusion(left, right, up, down, sharedImage, tidx, tidy);

	float previousColor = chebyshevPreviousImage[pixel];
	float inputData = sharedImage[tidy][tidx];
	jacobiOutput[pixel] = (omega * (gamma * (result - inputData) + inputData - previousColor)) + previousColor;
	chebyshevPreviousImage[pixel] = inputData;
	
}

void GPULoadWeights(float beta) {
	
	float weights[257];
	for(int w = 0; w < 256; w++) weights[w] = expf(-beta * w);
	weights[256] = 0;
	cudaMemcpyToSymbol(deviceWeights, weights, 257 * sizeof(float), 0, cudaMemcpyHostToDevice);
	GPUCheckError("GPULoadWeights");

}

void GPUMatrixFreeSolver(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, int level)
{

	//General parameters
	int iteration;
	
	//Jacobi + Chebyshev parameters
	int S = 10;
	float omega;
	float rho = 0.99;
	float gamma = 0.99;

	dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	
	cudaMemset(devicePreviousImage[level], 0, rows * cols * sizeof(float));
	copyFromPitchedData<<<grid, threads>>>(deviceNextImage[level], depthImage, depthPitch, rows, cols);
	copyFromPitchedData<<<grid, threads>>>(deviceDepthImage[level], depthImage, depthPitch, rows, cols);
	loadIndexToWeight<<<grid, threads>>>(grayImage, depthImage, deviceIndexToWeight[level], grayPitch, depthPitch, level, maxLevel, rows, cols);
	
	for(iteration = 0; iteration < maxIterations; iteration++) {
		
		if (iteration < S) omega = 1;
		else if (iteration == S) omega = 2.0 / (2.0 - rho * rho);
		else omega = 4.0 / (4.0 - rho * rho * omega);

		if (iteration % 2 == 0) {
			matrixFreeSolver << <grid, threads >> > (deviceDepthImage[level], deviceIndexToWeight[level], scribbleImage, deviceError[level], depthPitch, 
				scribblePitch, rows, cols, deviceNextImage[level], devicePreviousImage[level], omega, gamma);
		} else {
			matrixFreeSolver << <grid, threads >> > (deviceNextImage[level], deviceIndexToWeight[level], scribbleImage, deviceError[level], depthPitch, 
				scribblePitch, rows, cols, deviceDepthImage[level], devicePreviousImage[level], omega, gamma);
		}
			
	}
	
	if((iteration - 1) % 2 == 1) copyToPitchedData <<<grid, threads >>>(depthImage, deviceDepthImage[level], depthPitch, rows, cols);
	else copyToPitchedData <<<grid, threads >>>(depthImage, deviceNextImage[level], depthPitch, rows, cols);
	
	GPUCheckError("GPUMatrixFreeSolver");

}
