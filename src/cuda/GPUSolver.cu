//#define VIENNACL_WITH_CUDA
//#define PARALUTION 
//#define CUSP
#define TILE_WIDTH 16

#include "cuda/GPUSolver.h"
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#ifdef CUSP
	#include <cusp/ell_matrix.h>
	#include <cusp/csr_matrix.h>
	#include <cusp/monitor.h>
	#include <cusp/krylov/bicgstab.h>
	#include <cusp/relaxation/gauss_seidel.h>
	#include <cusp/relaxation/jacobi.h>
#endif
#ifdef PARALUTION
	#include <paralution.hpp>
#endif
#ifdef VIENNACL_WITH_CUDA
	#include <viennacl/compressed_matrix.hpp>
	#include <viennacl/ell_matrix.hpp>
	#include <viennacl/vector.hpp>
	#include <viennacl/linalg/bicgstab.hpp>
	#include <viennacl/linalg/jacobi_precond.hpp>
	#include <viennacl/linalg/ilu.hpp>
	#include <viennacl/linalg/amg.hpp>
#endif

//Global variables shared by the distinct solvers
float **devicePreviousImage;
float **deviceNextImage;
float **deviceDepthImage;
float **deviceError;
int **devicehorizontalIndexToWeight;
int **deviceverticalIndexToWeight;
int **deviceEdges;
int maxLevel;
__constant__ float deviceWeights[256];

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
	devicehorizontalIndexToWeight = (int**)malloc(sizeof(int*) * levels);
	deviceverticalIndexToWeight = (int**)malloc(sizeof(int*) * levels);
	deviceEdges = (int**)malloc(sizeof(int*) * levels);

	for(int level = 0; level < levels; level++) {
		int rowsPerLevel = rows / powf(2, level);
		int colsPerLevel = cols / powf(2, level);
		cudaMalloc((void**)&devicePreviousImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceNextImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceDepthImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceError[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&devicehorizontalIndexToWeight[level], sizeof(int) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceverticalIndexToWeight[level], sizeof(int) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceEdges[level], sizeof(int) * rowsPerLevel * colsPerLevel);
	}

	maxLevel = levels - 1;
#ifdef PARALUTION
	paralution::init_paralution();
#endif
	GPUCheckError("GPUAllocateDeviceMemory");
	
}

void GPUFreeDeviceMemory(int levels) {

	for(int level = 0; level < levels; level++) {
		cudaFree(devicePreviousImage[level]);
		cudaFree(deviceNextImage[level]);
		cudaFree(deviceDepthImage[level]);
		cudaFree(deviceError[level]);
		cudaFree(devicehorizontalIndexToWeight[level]);
		cudaFree(deviceverticalIndexToWeight[level]);
		cudaFree(deviceEdges[level]);
	}

#ifdef PARALUTION
	paralution::stop_paralution();
#endif
	GPUCheckError("GPUFreeDeviceMemory");

}

__device__ float solveDiffusion(int left, int right, int up, int down, float sharedImage[][TILE_WIDTH + 2], int tidx, int tidy) {

	float count = 0;
	float weight = 0;
	float sum = 0;
	if (left < 256) {
		weight = deviceWeights[left];
		sum += weight * sharedImage[tidy][tidx - 1];
		count += weight;
	}
	if (right < 256) {
		weight = deviceWeights[right];
		sum += weight * sharedImage[tidy][tidx + 1];
		count += weight;
	}
	if (up < 256) {
		weight = deviceWeights[up];
		sum += weight * sharedImage[tidy - 1][tidx];
		count += weight;
	}
	if (down < 256) {
		weight = deviceWeights[down];
		sum += weight * sharedImage[tidy + 1][tidx];
		count += weight;
	}

	return min(max(sum / count, 0.0), 255.0);

}

__global__ void copyFromPinnedData(float *output, float *input, size_t inputPitch, int rows, int cols) 
{
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	float *inputRow = (float*)((char*)input + y * inputPitch);
	int pixel = y * cols + x;
	output[pixel] = inputRow[x];

}

__global__ void copyToPinnedData(float *output, float *input, size_t outputPitch, int rows, int cols) 
{
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	float *outputRow = (float*)((char*)output + y * outputPitch);
	int pixel = y * cols + x;
	outputRow[x] = input[pixel];

}

__global__ void loadIndexToWeight(unsigned char *grayImage, float *depthImage, int *edges, int *horizontalIndexToWeight, 
	int *verticalIndexToWeight, size_t grayPitch, size_t depthPitch, int method, int level, int maxLevel, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;
	
	unsigned char *grayImageRow = grayImage + y * grayPitch;
	unsigned char *grayImageRowMinus = grayImage + (y - 1) * grayPitch;
	unsigned char *grayImageRowPlus = grayImage + (y + 1) * grayPitch;
	
	float *depthImageRow = (float*)((char*)depthImage + y * depthPitch);
	float *depthImageRowMinus = (float*)((char*)depthImage + (y - 1) * depthPitch);
	float *depthImageRowPlus = (float*)((char*)depthImage + (y + 1) * depthPitch);
	
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ unsigned char sharedGrayImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	__shared__ unsigned char sharedDepthImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	
	sharedGrayImage[tidy][tidx] = grayImageRow[x];
	if(tidx == 1) sharedGrayImage[tidy][0] = grayImageRow[x - 1];
	if(tidx == TILE_WIDTH) sharedGrayImage[tidy][TILE_WIDTH + 1] = grayImageRow[x + 1];
	if(tidy == 1) sharedGrayImage[0][tidx] = grayImageRowMinus[x];
	if(tidy == TILE_WIDTH) sharedGrayImage[TILE_WIDTH + 1][tidx] = grayImageRowPlus[x];
	
	if(method == 1) {
		
		sharedDepthImage[tidy][tidx] = depthImageRow[x];
		if(tidx == 1) sharedDepthImage[tidy][0] = depthImageRow[x - 1];
		if(tidx == TILE_WIDTH) sharedDepthImage[tidy][TILE_WIDTH + 1] = depthImageRow[x + 1];
		if(tidy == 1) sharedDepthImage[0][tidx] = depthImageRowMinus[x];
		if(tidy == TILE_WIDTH) sharedDepthImage[TILE_WIDTH + 1][tidx] = depthImageRowPlus[x];
	
	}
	
	__syncthreads();
	
	int directions[4] = {257, 257, 257, 257};
	
	if(method == 0 || (method == 1 && (level == maxLevel))) {
		
		unsigned char grayIntensity = sharedGrayImage[tidy][tidx];
		if(x - 1 >= 0) directions[0] = abs(grayIntensity - sharedGrayImage[tidy][tidx - 1]);
		if(x + 1 < cols) directions[1] = abs(grayIntensity - sharedGrayImage[tidy][tidx + 1]);
		if(y - 1 >= 0) directions[2] = abs(grayIntensity - sharedGrayImage[tidy - 1][tidx]);
		if(y + 1 < rows) directions[3] = abs(grayIntensity - sharedGrayImage[tidy + 1][tidx]);
		
	} else {
		
		unsigned char grayIntensity = sharedGrayImage[tidy][tidx];
		unsigned char depthIntensity = sharedDepthImage[tidy][tidx];
		int threshold = 4;
		if(level == 0) threshold = 0;
		if(x - 1 >= 0) {
			if((abs(depthIntensity - sharedDepthImage[tidy][tidx - 1]) > threshold)) directions[0] = abs(grayIntensity - sharedGrayImage[tidy][tidx - 1]);
			else directions[0] = 0;
		}
		if(x + 1 < cols) {
			if((abs(depthIntensity - sharedDepthImage[tidy][tidx + 1]) > threshold)) directions[1] = abs(grayIntensity - sharedGrayImage[tidy][tidx + 1]);
			else directions[1] = 0;
		}
		if(y - 1 >= 0) {
			if((abs(depthIntensity - sharedDepthImage[tidy - 1][tidx]) > threshold)) directions[2] = abs(grayIntensity - sharedGrayImage[tidy - 1][tidx]);
			else directions[2] = 0;
		}
		if(y + 1 < rows) {
			if((abs(depthIntensity - sharedDepthImage[tidy + 1][tidx]) > threshold)) directions[3] = abs(grayIntensity - sharedGrayImage[tidy + 1][tidx]);
			else directions[3] = 0;
		}
	
	}
	
	horizontalIndexToWeight[y * cols + x] = directions[0] * 1000 + directions[1];
	verticalIndexToWeight[y * cols + x] = directions[2] * 1000 + directions[3];

}

__global__ void matrixFreeSolver(float *input, int *horizontalIndexToWeight, int *verticalIndexToWeight, unsigned char *scribbleImage, 
	float *error, size_t inputPitch, size_t scribblePitch, int rows, int cols, int solverCode, float *jacobiOutput = 0, bool isDebugEnabled = false, 
	float *chebyshevPreviousImage = 0, int GSColor = 0, float omega = 0, float gamma = 0) 
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= cols || y >= rows) return;

	int pixel = y * cols + x;
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ float sharedImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	sharedImage[tidy][tidx] = input[pixel];
	if (tidx == 1) sharedImage[tidy][0] = input[pixel - 1];
	if (tidx == TILE_WIDTH) sharedImage[tidy][TILE_WIDTH + 1] = input[pixel + 1];
	if (tidy == 1) sharedImage[0][tidx] = input[(y - 1) * cols + x];
	if (tidy == TILE_WIDTH) sharedImage[TILE_WIDTH + 1][tidx] = input[(y + 1) * cols + x];
	__syncthreads();
	
	unsigned char *scribbleImageRow = scribbleImage + y * scribblePitch;
	if(scribbleImageRow[x] == 255) return;
	if(abs((x % 2) - (y % 2)) != GSColor && solverCode == 2) return;

	int index = horizontalIndexToWeight[pixel];
	int left = index / 1000;
	int right = index % 1000;
	index = verticalIndexToWeight[pixel];
	int up = index / 1000;
	int down = index % 1000;
	float result = solveDiffusion(left, right, up, down, sharedImage, tidx, tidy);

	//Jacobi = 0, Jacobi + Chebyshev = 1, Gauss-Seidel = 2 
	if(solverCode == 0) {
		jacobiOutput[pixel] = result;
	} else if(solverCode == 1) {
		float previousColor = chebyshevPreviousImage[pixel];
		float inputData = sharedImage[tidy][tidx];
		jacobiOutput[pixel] = (omega * (gamma * (result - inputData) + inputData - previousColor)) + previousColor;
		chebyshevPreviousImage[pixel] = inputData;
	} else if(solverCode == 2) {
		float depth = sharedImage[tidy][tidx];
		input[pixel] = omega * (result - depth) + depth;
	}

	if(isDebugEnabled) error[pixel] = abs(result - sharedImage[tidy][tidx]);
	

}

void GPULoadWeights(float beta) {
	
	float weights[256];
	for(int w = 0; w < 256; w++) weights[w] = expf(-beta * w);
	cudaMemcpyToSymbol(deviceWeights, weights, 256 * sizeof(float), 0, cudaMemcpyHostToDevice);
	GPUCheckError("GPULoadWeights");

}

void GPUMatrixFreeSolver(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, std::string solverName, std::string method, 
	int level, bool isDebugEnabled)
{

	//General parameters
	int iteration;
	float error;
	
	//Jacobi + Chebyshev parameters
	int S = 10;
	float omega;
	float rho = 0.99;
	float gamma = 0.99;

	//Gauss-Seidel parameters
	float theta = 1.5;
	int maxColors = 2;

	dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	
	if(solverName != "GPU-GaussSeidel") {
		cudaMemset(devicePreviousImage[level], 0, rows * cols * sizeof(float));
		copyFromPinnedData<<<grid, threads>>>(deviceNextImage[level], depthImage, depthPitch, rows, cols);
	}
	copyFromPinnedData<<<grid, threads>>>(deviceDepthImage[level], depthImage, depthPitch, rows, cols);
	
	int methodCode = (method == "Liao") ? 0 : 1;
	loadIndexToWeight<<<grid, threads>>>(grayImage, depthImage, deviceEdges[level], devicehorizontalIndexToWeight[level], 
		deviceverticalIndexToWeight[level], grayPitch, depthPitch, methodCode, level, maxLevel, rows, cols);
	
	for(iteration = 0; iteration < maxIterations; iteration++) {
		
		if(solverName == "GPU-Jacobi") {
		
			if (iteration % 2 == 0) {
				matrixFreeSolver<<<grid, threads>>>(deviceDepthImage[level], devicehorizontalIndexToWeight[level], deviceverticalIndexToWeight[level], 
					scribbleImage, deviceError[level], depthPitch, scribblePitch, rows, cols, 0, deviceNextImage[level], isDebugEnabled);
			} else {
				matrixFreeSolver<<<grid, threads>>>(deviceNextImage[level], devicehorizontalIndexToWeight[level], deviceverticalIndexToWeight[level], 
					scribbleImage, deviceError[level], depthPitch, scribblePitch, rows, cols, 0, deviceDepthImage[level], isDebugEnabled);
			}
			
		} else if(solverName == "GPU-Jacobi-Chebyshev") {

			if (iteration < S) omega = 1;
			else if (iteration == S) omega = 2.0 / (2.0 - rho * rho);
			else omega = 4.0 / (4.0 - rho * rho * omega);

			if (iteration % 2 == 0) {
				matrixFreeSolver << <grid, threads >> > (deviceDepthImage[level], devicehorizontalIndexToWeight[level], deviceverticalIndexToWeight[level],
					scribbleImage, deviceError[level], depthPitch, scribblePitch, rows, cols, 1, deviceNextImage[level], isDebugEnabled,
					devicePreviousImage[level], 0, omega, gamma);
			} else {
				matrixFreeSolver << <grid, threads >> > (deviceNextImage[level], devicehorizontalIndexToWeight[level], deviceverticalIndexToWeight[level],
					scribbleImage, deviceError[level], depthPitch, scribblePitch, rows, cols, 1, deviceDepthImage[level], isDebugEnabled,
					devicePreviousImage[level], 0, omega, gamma);
			}
			
		} else {

			for(int color = 0; color < maxColors; color++)
				matrixFreeSolver<<<grid, threads>>>(deviceDepthImage[level], devicehorizontalIndexToWeight[level], deviceverticalIndexToWeight[level], 
					scribbleImage, deviceError[level], depthPitch, scribblePitch, rows, cols, 2, 0, isDebugEnabled, 0, color, theta, 0);
		
		} 
		
		if(iteration % 100 == 0 && isDebugEnabled) {
			thrust::device_ptr<float> tptr = thrust::device_pointer_cast(deviceError[level]);
			error = thrust::reduce(tptr, tptr + rows * cols)/(rows * cols);
			if(error < tolerance) break;
		}
		
	}
	
	if((iteration - 1) % 2 == 1 || solverName == "GPU-GaussSeidel") copyToPinnedData <<<grid, threads >>>(depthImage, deviceDepthImage[level], depthPitch, rows, cols);
	else copyToPinnedData <<<grid, threads >>>(depthImage, deviceNextImage[level], depthPitch, rows, cols);
	
	if (isDebugEnabled) std::cout << "Iterations: " << iteration << " | Error: " << error << std::endl;

}

void CUSPMatrixFreeSolver(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols, 
	float beta, int maxIterations, float tolerance, std::string solverName, bool isDebugEnabled)
{
#ifdef CUSP
    cusp::csr_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5);
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	int counter = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			A.row_offsets[pixel] = counter;

			if (scribbleImage[pixel] == 255) {
				A.column_indices[counter] = pixel;
				A.values[counter] = 1;
				b[pixel] = depthImage[pixel];
				counter++;
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				A.column_indices[counter] = (y - 1) * cols + x;
				A.values[counter] = -weights[pixel * 4 + 2];
				counter++;
				sum += weights[pixel * 4 + 2];
			}
			if (x > 0) {
				A.column_indices[counter] = y * cols + x - 1;
				A.values[counter] = -weights[pixel * 4 + 0];
				counter++;
				sum += weights[pixel * 4 + 0];
			}
			if (x < cols - 1) {
				A.column_indices[counter] = y * cols + x + 1;
				A.values[counter] = -weights[pixel * 4 + 1];
				sum += weights[pixel * 4 + 1];
				counter++;
			}
			if (y < rows - 1) {
				A.column_indices[counter] = (y + 1) * cols + x;
				A.values[counter] = -weights[pixel * 4 + 3];
				counter++;
				sum += weights[pixel * 4 + 3];
			}
			
			A.column_indices[counter] = pixel;
			A.values[counter] = sum;
			counter++;

		}
	}
	A.row_offsets[rows * cols] = counter;

	cusp::array1d<float, cusp::device_memory> d_x(x);
	cusp::array1d<float, cusp::device_memory> d_b(b);
	
	cusp::csr_matrix<int, float, cusp::device_memory> d_A(A);
	cusp::array1d<float, cusp::device_memory> d_r(A.num_rows);
	cusp::multiply(d_A, d_x, d_r);
	cusp::blas::axpy(d_b, d_r, float(-1));
	
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, tolerance, false);
	
	if(solverName == "CUSP-Jacobi") {

		cusp::relaxation::jacobi<float, cusp::device_memory> M(d_A);
		while (!monitor.finished(d_r))
		{
			M(d_A, d_b, d_x);
			cusp::multiply(d_A, d_x, d_r);
			cusp::blas::axpy(d_b, d_r, float(-1));
			++monitor;
		}

	} else {

		cusp::relaxation::gauss_seidel<float, cusp::device_memory> M(d_A);
		while (!monitor.finished(d_r))
		{
			M(d_A, d_b, d_x);
			cusp::multiply(d_A, d_x, d_r);
			cusp::blas::axpy(d_b, d_r, float(-1));
			++monitor;
		}
	
	}

	if(isDebugEnabled) monitor.print();
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];
#else
	std::cout << "CUSP not supported" << std::endl;
#endif
}

void CUSPPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols, float beta, 
	int maxIterations, float tolerance, bool isDebugEnabled)
{

#ifdef CUSP
	cusp::ell_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5, 5);
	const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int pixel = y * cols + x;
			for(int w = 0; w < 5; w++) {
				A.column_indices(pixel, w) = X;
				A.values(pixel, w) = 0;
			}
		}
	}

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int counter = 0;
			int pixel = y * cols + x;
			if (scribbleImage[pixel] == 255) {
				A.column_indices(pixel, 0) = pixel;
				A.values(pixel, 0) = 1;
				b[pixel] = depthImage[pixel];
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				A.column_indices(pixel, counter) = (y - 1) * cols + x;
				A.values(pixel, counter) = -weights[pixel * 4 + 2];
				sum += weights[pixel * 4 + 2];
				counter++;
			}
			if (x > 0) {
				A.column_indices(pixel, counter) = y * cols + x - 1;
				A.values(pixel, counter) = -weights[pixel * 4 + 0];
				sum += weights[pixel * 4 + 0];;
				counter++;
			}
			if (x < cols - 1) {
				A.column_indices(pixel, counter) = y * cols + x + 1;
				A.values(pixel, counter) = -weights[pixel * 4 + 1];
				sum += weights[pixel * 4 + 1];
				counter++;
			}
			if (y < rows - 1) {
				A.column_indices(pixel, counter) = (y + 1) * cols + x;
				A.values(pixel, counter) = -weights[pixel * 4 + 3];
				sum += weights[pixel * 4 + 3];
				counter++;
			}
			
			A.column_indices(pixel, counter) = pixel;
			A.values(pixel, counter) = sum;

		}
	}

	cusp::ell_matrix<int, float, cusp::device_memory> d_A(A);
	cusp::array1d<float, cusp::device_memory> d_x(x);
	cusp::array1d<float, cusp::device_memory> d_b(b);
	cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_rows);
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, tolerance, false);
	cusp::krylov::bicgstab(d_A, d_x, d_b, monitor, M);
	
	if(isDebugEnabled) monitor.print();
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];
#else
	std::cout << "CUSP not supported" << std::endl;
#endif
}

void ParalutionPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols, float beta, 
	int maxIterations, float tolerance)
{

#ifdef PARALUTION
	float *h_x = (float*)malloc(rows * cols * sizeof(float));
	float *h_b = (float*)malloc(rows * cols * sizeof(float));
	int *row_offsets = (int*)malloc((rows * cols + 1) * sizeof(int));
	int *col = (int*)malloc(rows * cols * 5 * sizeof(int));
	float *val = (float*)malloc(rows * cols * 5 * sizeof(float));

	for (int pixel = 0; pixel < rows * cols; pixel++) {
		h_x[pixel] = depthImage[pixel];
		h_b[pixel] = 0;
	}

	int counter = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			row_offsets[pixel] = counter;

			if (scribbleImage[pixel] == 255) {
				col[counter] = pixel;
				val[counter] = 1;
				h_b[pixel] = depthImage[pixel];
				counter++;
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				col[counter] = (y - 1) * cols + x;
				val[counter] = -weights[pixel * 4 + 2];
				sum += weights[pixel * 4 + 2];
				counter++;
			}
			if (x > 0) {
				col[counter] = y * cols + x - 1;
				val[counter] = -weights[pixel * 4 + 0];
				sum += weights[pixel * 4 + 0];
				counter++;
			}
			if (x < cols - 1) {
				col[counter] = y * cols + x + 1;
				val[counter] = -weights[pixel * 4 + 1];
				sum += weights[pixel * 4 + 1];
				counter++;
			}
			if (y < rows - 1) {
				col[counter] = (y + 1) * cols + x;
				val[counter] = -weights[pixel * 4 + 3];
				sum += weights[pixel * 4 + 3];
				counter++;
			}
			
			col[counter] = pixel;
			val[counter] = sum;
			counter++;

		}
	}
	row_offsets[rows * cols] = counter;

	paralution::LocalMatrix<float> A;
	paralution::LocalVector<float> x;
	paralution::LocalVector<float> b;

	A.AllocateCSR("A", rows * cols * 5, rows * cols, rows * cols);
	A.CopyFromCSR(row_offsets, col, val);
	A.ConvertToELL();
	x.Allocate("x", A.get_nrow());
	x.SetDataPtr(&h_x, "vector", A.get_nrow());
	b.Allocate("b", A.get_nrow());
	b.SetDataPtr(&h_b, "vector", A.get_nrow());
	
	paralution::BiCGStab<paralution::LocalMatrix<float>, paralution::LocalVector<float>, float > ls;
	paralution::Jacobi<paralution::LocalMatrix<float>, paralution::LocalVector<float>, float > p;
	
	A.MoveToAccelerator();
	x.MoveToAccelerator();
	b.MoveToAccelerator();
	ls.MoveToAccelerator();
	
	ls.Init(tolerance, tolerance, 1e+10, maxIterations);
	ls.SetOperator(A);
	ls.SetPreconditioner(p);
	ls.Verbose(0);
	ls.Build();
	ls.Solve(b, &x);
	
	x.MoveToHost();
	x.LeaveDataPtr(&h_x);

	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = h_x[pixel];

	delete [] h_x;
	delete [] h_b;
	delete [] row_offsets;
	delete [] col;
	delete [] val;

	A.Clear();
	x.Clear();
	b.Clear();
	ls.Clear();
#else
	std::cout << "Paralution is not supported" << std::endl;
#endif

}

void ViennaCLPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, float *weights, int rows, int cols, float beta, 
	int maxIterations, float tolerance)
{

#ifdef VIENNACL_WITH_CUDA
	std::vector<float> h_x(rows * cols);
	std::vector<float> h_b(rows * cols);
	std::vector< std::map< unsigned int, float> > h_A(rows * cols);

	for (int pixel = 0; pixel < rows * cols; pixel++) {
		h_x[pixel] = depthImage[pixel];
		h_b[pixel] = 0;
	}

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			
			if (scribbleImage[pixel] == 255) {
				h_A[pixel][pixel] = 1;
				h_b[pixel] = depthImage[pixel];
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				h_A[pixel][(y - 1) * cols + x] = -weights[pixel * 4 + 2];
				sum += weights[pixel * 4 + 2];
			}
			if (x > 0) {
				h_A[pixel][y * cols + x - 1] = -weights[pixel * 4 + 0];
				sum += weights[pixel * 4 + 0];
			}
			if (x < cols - 1) {
				h_A[pixel][y * cols + x + 1] = -weights[pixel * 4 + 1];
				sum += weights[pixel * 4 + 1];
			}
			if (y < rows - 1) {
				h_A[pixel][(y + 1) * cols + x] = -weights[pixel * 4 + 3];
				sum += weights[pixel * 4 + 3];
			}
			
			h_A[pixel][pixel] = sum;

		}
	}

	viennacl::context ctx;
	viennacl::vector<float> x(rows * cols, ctx);
	viennacl::vector<float> b(rows * cols, ctx);
	viennacl::ell_matrix<float> A;
	viennacl::copy(h_x, x);
	viennacl::copy(h_b, b);
	viennacl::copy(h_A, A);
	
	viennacl::linalg::bicgstab_tag config(tolerance, maxIterations);
	config.abs_tolerance(tolerance);
	viennacl::linalg::bicgstab_solver<viennacl::vector<float> > solver(config);
	solver.set_initial_guess(x);
	x = solver(A, b);
	viennacl::copy(x, h_x);

	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = h_x[pixel];
#else
	std::cout << "ViennaCL not supported" << std::endl;
#endif	
}
