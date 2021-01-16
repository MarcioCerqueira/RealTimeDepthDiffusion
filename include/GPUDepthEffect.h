#ifndef GPU_DEPTH_EFFECT_H
#define GPU_DEPTH_EFFECT_H

void GPUSimulateDefocus(unsigned char *originalImage, size_t originalPitch, float *depthImage, size_t depthPitch, 
	unsigned char *artisticImage, size_t artisticPitch, int rows, int cols);
void GPUSimulateDesaturation(unsigned char *originalImage, size_t originalPitch, unsigned char *grayImage, size_t grayPitch, 
	float *depthImage, size_t depthPitch, unsigned char *artisticImage, size_t artisticPitch, int rows, int cols);
void GPUSimulateHaze(unsigned char *originalImage, size_t originalPitch, float *depthImage, size_t depthPitch, 
	unsigned char *artisticImage, size_t artisticPitch, int rows, int cols);
#endif