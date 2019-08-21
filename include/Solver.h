#ifndef SOLVER_H
#define SOLVER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseQR>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>

#include <Epetra_SerialComm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <AztecOO.h>
#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosEpetraAdapter.hpp>
#include <ml_epetra_preconditioner.h>

#include "LAHBPCG.h"

class Solver
{

public:

    Solver(int rows, int cols);
    ~Solver();
    
    void setBeta(float beta) { this->beta = beta; }
    void setErrorThreshold(float threshold) { this->threshold = threshold; }
    void setMaximumNumberOfIterations(int maxIterations) { this->maxIterations = maxIterations; }
	void setMaxLevel(int maxLevel) { this->maxLevel = maxLevel;  }
	void setMethod(std::string method) { this->method = method; }
    void enableDebug() { this->isDebugEnabled = true; }

    void runMatrixFreeSolver(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, std::string solverMethod);
    void runConjugateGradient(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols);
	void runAMG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols);
	void runLAHBPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols);
	
	void computeWeights(unsigned char *grayImage, unsigned char *depthImage, int level, int rows, int cols);
	void computePositions(int rows, int cols);
	float* getWeights() { return weights; }

private:

    cv::Mat debugImage;
	float *image;
	float *weights;
	int *positions;
	int rows;
    int cols;
    int maxIterations;
	int maxLevel;
    float threshold;
    float beta;
    bool isDebugEnabled;
	std::string method; 
	
};

#endif 