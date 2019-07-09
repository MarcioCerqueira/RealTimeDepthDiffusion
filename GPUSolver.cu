#define VIENNACL_WITH_CUDA
#define PARALUTION 
#define CUSP
#define TILE_WIDTH 16

#include "GPUSolver.h"
#include <iostream>
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
float **deviceWeights;
float **deviceError;
size_t *weightsPitch;

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
	deviceWeights = (float**)malloc(sizeof(float*) * levels);
	deviceError = (float**)malloc(sizeof(float*) * levels);
	weightsPitch = (size_t*)malloc(sizeof(size_t*) * levels);
	
	for(int level = 0; level < levels; level++) {
		int rowsPerLevel = rows / powf(2, level);
		int colsPerLevel = rows / powf(2, level);
		cudaMalloc((void**)&devicePreviousImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMalloc((void**)&deviceNextImage[level], sizeof(float) * rowsPerLevel * colsPerLevel);
		cudaMallocPitch((void**)&deviceWeights[level], &weightsPitch[level], colsPerLevel * 4 * sizeof(float), rowsPerLevel);
		cudaMalloc((void**)&deviceError[level], sizeof(float) * rowsPerLevel * colsPerLevel);
	}

#ifdef PARALUTION
	paralution::init_paralution();
#endif
	GPUCheckError("GPUAllocateDeviceMemory");
	
}

void GPUFreeDeviceMemory(int levels) {

	for(int level = 0; level < levels; level++) {
		cudaFree(devicePreviousImage[level]);
		cudaFree(deviceNextImage[level]);
		cudaFree(deviceWeights[level]);
		cudaFree(deviceError[level]);
	}

#ifdef PARALUTION
	paralution::stop_paralution();
#endif
	GPUCheckError("GPUFreeDeviceMemory");

}

__global__ void convert(unsigned char *src, size_t srcPitch, float *dst, size_t dstPitch, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	float *dstRow = (float*)((char*)dst + y * dstPitch);
	unsigned char *srcRow = src + y * srcPitch;
	dstRow[x] = srcRow[x * 3 + 0];

}

__global__ void pyrDown(unsigned char *prevScribbleImage, size_t prevScribblePitch, unsigned char *prevEditedImage, size_t prevEditedPitch, 
	int previousRows, int previousCols, unsigned char *currScribbleImage, size_t currScribblePitch, unsigned char *currEditedImage, 
	size_t currEditedPitch, int currentRows, int currentCols)
{
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= currentCols || y >= currentRows) return;
    int kernelSize = 2;
    for(int py = 2 * y - kernelSize/2; py < 2 * y + kernelSize/2; py++) {
		for(int px = 2 * x - kernelSize/2; px < 2 * x + kernelSize/2; px++) {
			int pixel = py * previousCols + px;
            if(pixel >= 0 && pixel < previousCols * previousRows) {
				unsigned char *prevScribbleImageRow = prevScribbleImage + py * prevScribblePitch;
				if(prevScribbleImageRow[px] == 255) {
					unsigned char *currScribbleImageRow = currScribbleImage + y * currScribblePitch;
					unsigned char *currEditedImageRow = currEditedImage + y * currEditedPitch;
					unsigned char *prevEditedImageRow = prevEditedImage + py * prevEditedPitch;
					currScribbleImageRow[x] = 255;
                    currEditedImageRow[x * 3 + 0] =  prevEditedImageRow[px * 3 + 0];
                }           
			}
        }
    }
	
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

__global__ void computeWeights(unsigned char *grayImage, float *weights, size_t weightsPitch, size_t grayPitch, float beta, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;
	
	float *weightsRow = (float*)((char*)weights + y * weightsPitch);
	unsigned char *grayImageRow = grayImage + y * grayPitch;
	unsigned char *grayImageRowMinus = grayImage + (y - 1) * grayPitch;
	unsigned char *grayImageRowPlus = grayImage + (y + 1) * grayPitch;
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ unsigned char sharedGrayImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	sharedGrayImage[tidy][tidx] = grayImageRow[x];
	if(tidx == 1) sharedGrayImage[tidy][0] = grayImageRow[x - 1];
	if(tidx == TILE_WIDTH) sharedGrayImage[tidy][TILE_WIDTH + 1] = grayImageRow[x + 1];
	if(tidy == 1) sharedGrayImage[0][tidx] = grayImageRowMinus[x];
	if(tidy == TILE_WIDTH) sharedGrayImage[TILE_WIDTH + 1][tidx] = grayImageRowPlus[x];
	__syncthreads();
	
	unsigned char grayIntensity = sharedGrayImage[tidy][tidx];
    if(x - 1 >= 0) weightsRow[x * 4 + 0] = expf(-beta * abs(grayIntensity - sharedGrayImage[tidy][tidx - 1]));
    if(x + 1 < cols) weightsRow[x * 4 + 1] = expf(-beta * abs(grayIntensity - sharedGrayImage[tidy][tidx + 1]));
    if(y - 1 >= 0) weightsRow[x * 4 + 2] = expf(-beta * abs(grayIntensity - sharedGrayImage[tidy - 1][tidx]));
    if(y + 1 < rows) weightsRow[x * 4 + 3] = expf(-beta * abs(grayIntensity - sharedGrayImage[tidy + 1][tidx]));

}

__global__ void jacobi(float *output, float *input, float *weights, unsigned char *scribbleImage, float *error, size_t inputPitch, 
	size_t weightsPitch, size_t scribblePitch, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int pixel = y * cols + x;
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ float sharedImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	float *inputRow = (float*)((char*)input + y * inputPitch);
	float *inputRowMinus = (float*)((char*)input + (y - 1) * inputPitch);
	float *inputRowPlus = (float*)((char*)input + (y + 1) * inputPitch);
	sharedImage[tidy][tidx] = inputRow[x];
	if(tidx == 1) sharedImage[tidy][0] = inputRow[x - 1];
	if(tidx == TILE_WIDTH) sharedImage[tidy][TILE_WIDTH + 1] = inputRow[x + 1];
	if(tidy == 1) sharedImage[0][tidx] = inputRowMinus[x];
	if(tidy == TILE_WIDTH) sharedImage[TILE_WIDTH + 1][tidx] = inputRowPlus[x];
	__syncthreads();
	
	unsigned char *scribbleImageRow = scribbleImage + y * scribblePitch;
	if(scribbleImageRow[x] == 255) return;
	            
    float count = 0;
    float sum = 0;
	float weight = 0;
	float *weightsRow = (float*)((char*)weights + y * weightsPitch);
	
	if(x - 1 >= 0) {
		weight = weightsRow[x * 4 + 0];
		sum += weight * sharedImage[tidy][tidx - 1];
		count += weight;
	}
    if(x + 1 < cols) {
		weight = weightsRow[x * 4 + 1];
		sum += weight * sharedImage[tidy][tidx + 1];
		count += weight;
	}
    if(y - 1 >= 0) {
		weight = weightsRow[x * 4 + 2];
		sum += weight * sharedImage[tidy - 1][tidx];
		count += weight;
	}
    if(y + 1 < rows) {
		weight = weightsRow[x * 4 + 3];
		sum += weight * sharedImage[tidy + 1][tidx];
		count += weight;
	}

	if(count > 0) {
		output[pixel] = sum/count;
		error[pixel] = abs(sum/count - sharedImage[tidy][tidx]);
	} else {
		error[pixel] = 0;
	}

}

__global__ void chebyshevSemiIterativeMethod(float *previousImage, float *output, float *input, float *weights, unsigned char *scribbleImage, 
	float *error, size_t inputPitch, size_t weightsPitch, size_t scribblePitch, float omega, float gamma, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int pixel = y * cols + x;
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ float sharedImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	float *inputRow = (float*)((char*)input + y * inputPitch);
	float *inputRowMinus = (float*)((char*)input + (y - 1) * inputPitch);
	float *inputRowPlus = (float*)((char*)input + (y + 1) * inputPitch);
	sharedImage[tidy][tidx] = inputRow[x];
	if(tidx == 1) sharedImage[tidy][0] = inputRow[x - 1];
	if(tidx == TILE_WIDTH) sharedImage[tidy][TILE_WIDTH + 1] = inputRow[x + 1];
	if(tidy == 1) sharedImage[0][tidx] = inputRowMinus[x];
	if(tidy == TILE_WIDTH) sharedImage[TILE_WIDTH + 1][tidx] = inputRowPlus[x];
	__syncthreads();
	
	unsigned char *scribbleImageRow = scribbleImage + y * scribblePitch;
	if(scribbleImageRow[x] == 255) return;
	            
    float count = 0;
    float sum = 0;
	float weight = 0;
	float *weightsRow = (float*)((char*)weights + y * weightsPitch);
	
	if(x - 1 >= 0) {
		weight = weightsRow[x * 4 + 0];
		sum += weight * sharedImage[tidy][tidx - 1];
		count += weight;
	}
    if(x + 1 < cols) {
		weight = weightsRow[x * 4 + 1];
		sum += weight * sharedImage[tidy][tidx + 1];
		count += weight;
	}
    if(y - 1 >= 0) {
		weight = weightsRow[x * 4 + 2];
		sum += weight * sharedImage[tidy - 1][tidx];
		count += weight;
	}
    if(y + 1 < rows) {
		weight = weightsRow[x * 4 + 3];
		sum += weight * sharedImage[tidy + 1][tidx];
		count += weight;
	}

	if(count > 0) {
		float previousColor = previousImage[pixel];
		output[pixel] = (omega * (gamma * (sum/count - sharedImage[tidy][tidx]) + sharedImage[tidy][tidx] - previousColor)) + previousColor;
		previousImage[pixel] = sharedImage[tidy][tidx];
		error[pixel] = abs(sum/count - sharedImage[tidy][tidx]);
	} else {
		error[pixel] = 0;
	}

}

__global__ void gaussSeidel(float *image, float *weights, unsigned char *scribbleImage, float *error, size_t imagePitch, size_t weightsPitch, 
	size_t scribblePitch, int color, float omega, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int pixel = y * cols + x;
	int tidx = threadIdx.x + 1;
	int tidy = threadIdx.y + 1;
	
	__shared__ float sharedImage[TILE_WIDTH + 2][TILE_WIDTH + 2];
	float *imageRow = (float*)((char*)image + y * imagePitch);
	float *imageRowMinus = (float*)((char*)image + (y - 1) * imagePitch);
	float *imageRowPlus = (float*)((char*)image + (y + 1) * imagePitch);
	sharedImage[tidy][tidx] = imageRow[x];
	if(tidx == 1) sharedImage[tidy][0] = imageRow[x - 1];
	if(tidx == TILE_WIDTH) sharedImage[tidy][TILE_WIDTH + 1] = imageRow[x + 1];
	if(tidy == 1) sharedImage[0][tidx] = imageRowMinus[x];
	if(tidy == TILE_WIDTH) sharedImage[TILE_WIDTH + 1][tidx] = imageRowPlus[x];
	__syncthreads();
	
	unsigned char *scribbleImageRow = scribbleImage + y * scribblePitch;
	if(scribbleImageRow[x] == 255) return;
	if(abs((x % 2) - (y % 2)) != color) return;
	            
    float count = 0;
    float sum = 0;
	float weight = 0;
	float *weightsRow = (float*)((char*)weights + y * weightsPitch);
	
	if(x - 1 >= 0) {
		weight = weightsRow[x * 4 + 0];
		sum += weight * sharedImage[tidy][tidx - 1];
		count += weight;
	}
    if(x + 1 < cols) {
		weight = weightsRow[x * 4 + 1];
		sum += weight * sharedImage[tidy][tidx + 1];
		count += weight;
	}
    if(y - 1 >= 0) {
		weight = weightsRow[x * 4 + 2];
		sum += weight * sharedImage[tidy - 1][tidx];
		count += weight;
	}
    if(y + 1 < rows) {
		weight = weightsRow[x * 4 + 3];
		sum += weight * sharedImage[tidy + 1][tidx];
		count += weight;
	}

	if(count > 0) {
		error[pixel] = abs(sum/count - sharedImage[tidy][tidx]);
		float depth = sharedImage[tidy][tidx];
		imageRow[x] = omega * (sum / count - depth) + depth;
	} else {
		error[pixel] = 0;
	}

}

void GPUConvertToFloat(unsigned char *src, size_t srcPitch, float *dst, size_t dstPitch, int rows, int cols)
{
	dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	convert<<<grid, threads>>>(src, srcPitch, dst, dstPitch, rows, cols);
}

void GPUPyrDownAnnotation(unsigned char *prevScribbleImage, size_t prevScribblePitch, unsigned char *prevEditedImage, size_t prevEditedPitch, 
	int previousRows, int previousCols, unsigned char *currScribbleImage, size_t currScribblePitch, unsigned char *currEditedImage, 
	size_t currEditedPitch, int currentRows, int currentCols)
{

	dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(divUp(currentCols, threads.x), divUp(currentRows, threads.y));
	pyrDown<<<grid, threads>>>(prevScribbleImage, prevScribblePitch, prevEditedImage, prevEditedPitch, previousRows, previousCols, 
		currScribbleImage, currScribblePitch, currEditedImage, currEditedPitch, currentRows, currentCols);

}

void GPUJacobi(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled, bool chebyshevVariant, 
	int level)
{

	int iteration;
	float error;
	//Chebyshev's variant
	int S = 10;
	double omega;
	double rho = 0.99;
	double gamma = 0.99;
	
	dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	
	cudaMemset(devicePreviousImage[level], 0, rows * cols * sizeof(float));
	copyFromPinnedData<<<grid, threads>>>(deviceNextImage[level], depthImage, depthPitch, rows, cols);
	computeWeights<<<grid, threads>>>(grayImage, deviceWeights[level], weightsPitch[level], grayPitch, beta, rows, cols);
	
	for(iteration = 0; iteration < maxIterations; iteration++) {
		
		if(!chebyshevVariant) {
		
			jacobi<<<grid, threads>>>(deviceNextImage[level], depthImage, deviceWeights[level], scribbleImage, deviceError[level], 
				depthPitch, weightsPitch[level], scribblePitch, rows, cols);
		
		} else {
			
			if (iteration < S) omega = 1;
			else if (iteration == S) omega = 2.0 / (2.0 - rho * rho);
			else omega = 4.0 / (4.0 - rho * rho * omega);

			chebyshevSemiIterativeMethod<<<grid, threads>>>(devicePreviousImage[level], deviceNextImage[level], depthImage, 
				deviceWeights[level], scribbleImage, deviceError[level], depthPitch, weightsPitch[level], scribblePitch, omega, gamma, rows, cols);

		} 
		
		copyToPinnedData<<<grid, threads>>>(depthImage, deviceNextImage[level], depthPitch, rows, cols);
    
		if(iteration % 100 == 0) {
			thrust::device_ptr<float> tptr = thrust::device_pointer_cast(deviceError[level]);
			error = thrust::reduce(tptr, tptr + rows * cols)/(rows * cols);
			if(error < tolerance) break;
		}
		
	}

	if (isDebugEnabled) std::cout << "Iterations: " << iteration << " | Error: " << error << std::endl;

}

void GPUGaussSeidel(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled, int level)
{

	int iteration;
	float error;

	float omega = 1.9;
	
	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	int maxColors = 2;

	computeWeights<<<grid, threads>>>(grayImage, deviceWeights[level], weightsPitch[level], grayPitch, beta, rows, cols);
	for(iteration = 0; iteration < maxIterations; iteration++) {

		for(int color = 0; color < maxColors; color++)
			gaussSeidel<<<grid, threads>>>(depthImage, deviceWeights[level], scribbleImage, deviceError[level], depthPitch, weightsPitch[level], 
				scribblePitch, color, omega, rows, cols);
	
		if(iteration % 100 == 0) {
			thrust::device_ptr<float> tptr = thrust::device_pointer_cast(deviceError[level]);
			error = thrust::reduce(tptr, tptr + rows * cols)/(rows * cols);
			if(error < tolerance) break;
		}

	}
	if (isDebugEnabled) std::cout << "Iterations: " << iteration << " | Error: " << error << std::endl;

}

void CUSPJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	float beta, int maxIterations, float tolerance, bool isDebugEnabled)
{
#ifdef CUSP
    cusp::csr_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5);
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	float weight;
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
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				A.column_indices[counter] = (y - 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				A.column_indices[counter] = y * cols + x - 1;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				A.column_indices[counter] = y * cols + x + 1;
				A.values[counter] = -weight;
				sum += weight;
				counter++;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				A.column_indices[counter] = (y + 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
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
	cusp::relaxation::jacobi<float, cusp::device_memory> M(d_A);
	cusp::array1d<float, cusp::device_memory> d_r(A.num_rows);
	cusp::multiply(d_A, d_x, d_r);
	cusp::blas::axpy(d_b, d_r, float(-1));
	
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, tolerance, false);
	
	while (!monitor.finished(d_r))
	{
		M(d_A, d_b, d_x);
		cusp::multiply(d_A, d_x, d_r);
		cusp::blas::axpy(d_b, d_r, float(-1));
		++monitor;
	}
	
	if(isDebugEnabled) monitor.print();
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];
#else
	std::cout << "CUSP not supported" << std::endl;
#endif
}

void CUSPGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	float beta, int maxIterations, float tolerance, bool isDebugEnabled)
{
#ifdef CUSP
    cusp::csr_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5);
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	float weight;
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
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				A.column_indices[counter] = (y - 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				A.column_indices[counter] = y * cols + x - 1;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				A.column_indices[counter] = y * cols + x + 1;
				A.values[counter] = -weight;
				sum += weight;
				counter++;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				A.column_indices[counter] = (y + 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
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
	cusp::relaxation::gauss_seidel<float, cusp::device_memory> M(d_A);
	cusp::array1d<float, cusp::device_memory> d_r(A.num_rows);
	cusp::multiply(d_A, d_x, d_r);
	cusp::blas::axpy(d_b, d_r, float(-1));
	
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, tolerance, false);
	
	while (!monitor.finished(d_r))
	   {
		   M(d_A, d_b, d_x);
		   cusp::multiply(d_A, d_x, d_r);
		   cusp::blas::axpy(d_b, d_r, float(-1));
		   ++monitor;
	   }
	
	if(isDebugEnabled) monitor.print();
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];
#else
	std::cout << "CUSP not supported" << std::endl;
#endif
}

void CUSPPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, float beta, 
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

	float weight;
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
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				A.column_indices(pixel, counter) = (y - 1) * cols + x;
				A.values(pixel, counter) = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				A.column_indices(pixel, counter) = y * cols + x - 1;
				A.values(pixel, counter) = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				A.column_indices(pixel, counter) = y * cols + x + 1;
				A.values(pixel, counter) = -weight;
				sum += weight;
				counter++;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				A.column_indices(pixel, counter) = (y + 1) * cols + x;
				A.values(pixel, counter) = -weight;
				counter++;
				sum += weight;
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

void ParalutionPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, float beta, 
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

	float weight;
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
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				col[counter] = (y - 1) * cols + x;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				col[counter] = y * cols + x - 1;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				col[counter] = y * cols + x + 1;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				col[counter] = (y + 1) * cols + x;
				val[counter] = -weight;
				counter++;
				sum += weight;
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

void ViennaCLPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, float beta, 
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

	float weight;
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
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				h_A[pixel][(y - 1) * cols + x] = -weight;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				h_A[pixel][y * cols + x - 1] = -weight;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				h_A[pixel][y * cols + x + 1] = -weight;
				sum += weight;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				h_A[pixel][(y + 1) * cols + x] = -weight;
				sum += weight;
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
