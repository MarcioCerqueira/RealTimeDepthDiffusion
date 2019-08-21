#include "cuda/GPUDepthEffect.h"
#include <cuda_runtime.h>

int divUp2(int a, int b) { 
    return (a + b - 1)/b;
}

__global__ void simulateDesaturation(unsigned char *originalImage, unsigned char *grayImage, float *depthImage, unsigned char *artisticImage, 
	size_t originalPitch, size_t grayPitch, size_t depthPitch, size_t artisticPitch, int rows, int cols)
{
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= cols || y >= rows) return;
	
	unsigned char *originalImageRow = originalImage + y * originalPitch;
	unsigned char *grayImageRow = grayImage + y * grayPitch;
	unsigned char *artisticImageRow = artisticImage + y * artisticPitch;
	float *depthImageRow = (float*)((char*)depthImage + y * depthPitch);
	
	float f = depthImageRow[x] / 255.0;
	artisticImageRow[x * 3 + 0] = f * grayImageRow[x] + (1 - f) * originalImageRow[x * 3 + 0];
	artisticImageRow[x * 3 + 1] = f * grayImageRow[x] + (1 - f) * originalImageRow[x * 3 + 1];
	artisticImageRow[x * 3 + 2] = f * grayImageRow[x] + (1 - f) * originalImageRow[x * 3 + 2];

}

__global__ void simulateDefocus(unsigned char *originalImage, float *depthImage, unsigned char *artisticImage, size_t originalPitch, 
	size_t depthPitch, size_t artisticPitch, int rows, int cols)
{
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= cols || y >= rows) return;
	
	unsigned char *originalImageRow = originalImage + y * originalPitch;
	unsigned char *artisticImageRow = artisticImage + y * artisticPitch;
	float *depthImageRow = (float*)((char*)depthImage + y * depthPitch);
	
	int kernelSize = 0.025 * sqrtf(rows * rows + cols * cols);
	int anisotropicKernelSize = kernelSize * depthImageRow[x] / 255.0;
	float sum[3] = {0, 0, 0};
	int count = 0;

	for (int py = y - anisotropicKernelSize / 2; py < y + anisotropicKernelSize / 2; py++) {
		for (int px = x - anisotropicKernelSize / 2; px < x + anisotropicKernelSize / 2; px++) {

			if (px >= 0 && py >= 0 && px < cols && py < rows) {

				unsigned char *currentOriginalImageRow = originalImage + py * originalPitch;
				sum[0] += currentOriginalImageRow[px * 3 + 0];
				sum[1] += currentOriginalImageRow[px * 3 + 1];
				sum[2] += currentOriginalImageRow[px * 3 + 2];
				count++;

			}
		}
	}

	if (count == 0) {
		artisticImageRow[x * 3 + 0] = originalImageRow[x * 3 + 0];
		artisticImageRow[x * 3 + 1] = originalImageRow[x * 3 + 1];
		artisticImageRow[x * 3 + 2] = originalImageRow[x * 3 + 2];
	} else {
		artisticImageRow[x * 3 + 0] = sum[0] / count;
		artisticImageRow[x * 3 + 1] = sum[1] / count;
		artisticImageRow[x * 3 + 2] = sum[2] / count;
	}

}

__global__ void simulateHaze(unsigned char *originalImage, float *depthImage, unsigned char *artisticImage, size_t originalPitch, 
	size_t depthPitch, size_t artisticPitch, int rows, int cols)
{
	
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= cols || y >= rows) return;
	
	unsigned char *originalImageRow = originalImage + y * originalPitch;
	unsigned char *artisticImageRow = artisticImage + y * artisticPitch;
	float *depthImageRow = (float*)((char*)depthImage + y * depthPitch);
	
	float beta = 2;
	float t = expf(-beta * depthImageRow[x] / 255.0);
	artisticImageRow[x * 3 + 0] = t * originalImageRow[x * 3 + 0] + (1 - t) * 255;
	artisticImageRow[x * 3 + 1] = t * originalImageRow[x * 3 + 1] + (1 - t) * 255;
	artisticImageRow[x * 3 + 2] = t * originalImageRow[x * 3 + 2] + (1 - t) * 255;

}

void GPUSimulateDesaturation(unsigned char *originalImage, size_t originalPitch, unsigned char *grayImage, size_t grayPitch, 
	float *depthImage, size_t depthPitch, unsigned char *artisticImage, size_t artisticPitch, int rows, int cols) 
{

	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	simulateDesaturation<<<grid, threads>>>(originalImage, grayImage, depthImage, artisticImage, originalPitch, grayPitch, depthPitch, artisticPitch, rows, cols);

}

void GPUSimulateDefocus(unsigned char *originalImage, size_t originalPitch, float *depthImage, size_t depthPitch, 
	unsigned char *artisticImage, size_t artisticPitch, int rows, int cols)
{

	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	simulateDefocus<<<grid, threads>>>(originalImage, depthImage, artisticImage, originalPitch, depthPitch, artisticPitch, rows, cols);

}

void GPUSimulateHaze(unsigned char *originalImage, size_t originalPitch, float *depthImage, size_t depthPitch, unsigned char *artisticImage, 
	size_t artisticPitch, int rows, int cols) 
{
	
	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	simulateHaze<<<grid, threads>>>(originalImage, depthImage, artisticImage, originalPitch, depthPitch, artisticPitch, rows, cols);

}