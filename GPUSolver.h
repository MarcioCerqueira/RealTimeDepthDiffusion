#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

void GPUAllocateDeviceMemory(int rows, int cols, int levels);
void GPUFreeDeviceMemory(int levels);
void GPULoadWeights(float beta);
void GPUConvertToFloat(unsigned char *src, size_t srcPitch, float *dst, size_t dstPitch, int rows, int cols);
void GPUPyrDownAnnotation(unsigned char *prevScribbleImage, size_t prevScribblePitch, unsigned char *prevEditedImage, 
	size_t prevEditedPitch, int previousRows, int previousCols, unsigned char *currScribbleImage, size_t currScribblePitch, 
	unsigned char *currEditedImage, size_t currEditedPitch, int currentRows, int currentCols);
void GPUJacobi(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, unsigned char *grayImage, 
	size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled, bool chebyshevVariant, 
	int level);
void GPUGaussSeidel(float *depthImage, size_t depthPitch, unsigned char *scribbleImage, size_t scribblePitch, 
	unsigned char *grayImage, size_t grayPitch, int rows, int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled, 
	int level);
void CUSPJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled);
void CUSPGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled);
void CUSPPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance, bool isDebugEnabled);
void ParalutionPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void ViennaCLPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);

#endif