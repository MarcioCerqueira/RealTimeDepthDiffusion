#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

void GPUJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void GPUGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void CUSPJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void CUSPGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void CUSPPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void ParalutionPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);
void ViennaCLPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows,
	int cols, float beta, int maxIterations, float tolerance);

#endif