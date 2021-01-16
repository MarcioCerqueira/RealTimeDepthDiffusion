#include "GPUImageProcessing.h"
#include <cuda_runtime.h>

int divUp3(int a, int b) { 
    return (a + b - 1)/b;
}

__global__ void convert(unsigned char *src, size_t srcPitch, float *dst, size_t dstPitch, unsigned char *mask, size_t maskPitch, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	float *dstRow = (float*)((char*)dst + y * dstPitch);
	unsigned char *srcRow = src + y * srcPitch;
	unsigned char *maskRow = mask + y * maskPitch;
	if(maskRow[x] == 255) dstRow[x] = srcRow[x * 3 + 0];

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
            if(px >= 0 && py >= 0 && px < previousCols && py < previousRows) {
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

__global__ void paintImage(int x, int y, int scribbleColor, int scribbleRadius, unsigned char *editedImage, size_t editedPitch, 
	unsigned char *scribbleImage, size_t scribblePitch, int rows, int cols)
{
	
	const int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const int threadY = blockDim.y * blockIdx.y + threadIdx.y;

	if(threadX >= cols || threadY >= rows) return;
	if(threadX < x - scribbleRadius/2 || threadX > x + scribbleRadius/2) return;
	if(threadY < y - scribbleRadius/2 || threadY > y + scribbleRadius/2) return;

	unsigned char *editedImageRow = editedImage + threadY * editedPitch;
	unsigned char *scribbleImageRow = scribbleImage + threadY * scribblePitch;

	editedImageRow[threadX * 3 + 0] = scribbleColor;
	editedImageRow[threadX * 3 + 1] = scribbleColor;
	editedImageRow[threadX * 3 + 2] = scribbleColor;
	scribbleImageRow[threadX] = 255;

}

void GPUConvertToFloat(unsigned char *src, size_t srcPitch, float *dst, size_t dstPitch, unsigned char *mask, size_t maskPitch, int rows, int cols)
{

	dim3 threads(16, 16);
    dim3 grid(divUp3(cols, threads.x), divUp3(rows, threads.y));
	convert<<<grid, threads>>>(src, srcPitch, dst, dstPitch, mask, maskPitch, rows, cols);

}

void GPUPyrDownAnnotation(unsigned char *prevScribbleImage, size_t prevScribblePitch, unsigned char *prevEditedImage, size_t prevEditedPitch, 
	int previousRows, int previousCols, unsigned char *currScribbleImage, size_t currScribblePitch, unsigned char *currEditedImage, 
	size_t currEditedPitch, int currentRows, int currentCols)
{

	dim3 threads(16, 16);
    dim3 grid(divUp3(currentCols, threads.x), divUp3(currentRows, threads.y));
	pyrDown<<<grid, threads>>>(prevScribbleImage, prevScribblePitch, prevEditedImage, prevEditedPitch, previousRows, previousCols, 
		currScribbleImage, currScribblePitch, currEditedImage, currEditedPitch, currentRows, currentCols);

}

void GPUPaintImage(int x, int y, int scribbleColor, int scribbleRadius, unsigned char *editedImage, size_t editedPitch, 
	unsigned char *scribbleImage, size_t scribblePitch, int rows, int cols) 
{

	dim3 threads(16, 16);
    dim3 grid(divUp3(cols, threads.x), divUp3(rows, threads.y));
	paintImage<<<grid, threads>>>(x, y, scribbleColor, scribbleRadius, editedImage, editedPitch, scribbleImage, scribblePitch, rows, cols);

}