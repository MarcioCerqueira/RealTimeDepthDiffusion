#ifndef SOLVER_H
#define SOLVER_H

#include <opencv2\opencv.hpp>
#include <Eigen\IterativeLinearSolvers>
#include <cstdlib>
#include <cmath>
#include <iostream>

class Solver
{

public:

    Solver(int rows, int cols);
    ~Solver();
    
    
    void setBeta(float beta) { this->beta = beta; }
    void setErrorThreshold(float threshold) { this->threshold = threshold; }
    void setMaximumNumberOfIterations(int maxIterations) { this->maxIterations = maxIterations; }
    void enableDebug() { this->isDebugEnabled = true; }

    void runJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols);
    void runGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols);
    void runConjugateGradient(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols);

private:

    float *image;
    cv::Mat debugImage;
    int rows;
    int cols;
    int maxIterations;
    float threshold;
    float beta;
    bool isDebugEnabled;

};

#endif 