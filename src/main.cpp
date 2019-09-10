/*
References:
	Locally Adapted Hierarchical Basis Preconditioning by R. Szeliski ToG 2006
		Source code was directly adapted from "https://github.com/s-gupta/rgbdutils/blob/master/imagestack/src/LAHBPCG.cpp"
	Diffusion Curves - "Diffusion Curves: A Vector Representation for Smooth-Shaded Images" by Orzan et al. ToG 2008
	Haze Simulation - Equations (1, 2) of "Single Image Haze Removal using Dark Channel Prior" by He et al. CVPR 2009
    Chebyshev Semi-Iterative Method - "A Chebyshev Semi-Iterative Approach for Accelerating Projective and PBD" by H. Wang ToG 2015 
    Depth Design - "Depth Annotations: Designing Depth of a Single Image for Depth-based Effects" by Liao et al. GI 2017
	GPU-based Red-Black Gauss-Seidel - "Fast Quadrangular Mass-Spring Systems using Red-Black Ordering" by Pall et al. VRIPHYS 2018
*/

#define OPENCV_WITH_CUDA

#include "Solver.h"
#include "DepthEffect.h"
#include "cuda/GPUSolver.h"
#include "cuda/GPUDepthEffect.h"
#include "cuda/GPUImageProcessing.h"
#include <vector>
#include <opencv2/opencv.hpp>
#ifdef OPENCV_WITH_CUDA
	#include <opencv2/gpu/gpu.hpp>
	using namespace cv::gpu;
#endif

bool buttonIsPressed = false;
int key = 0;
int scribbleColor = 0;
int scribbleRadius;
std::vector<cv::Mat> editedImage;
std::vector<cv::Mat> scribbleImage;
#ifdef OPENCV_WITH_CUDA
	std::vector<GpuMat> deviceEditedImage;
	std::vector<GpuMat> deviceScribbleImage;
#endif
std::string solverName = "GPU-Jacobi-Chebyshev";

double cpuTime(void)
{

	double value;
	value = (double) clock () / (double) CLOCKS_PER_SEC;
	return value;

}

void pyrDownAnnotation(std::vector<cv::Mat>& editedImage, std::vector<cv::Mat>& scribbleImage, int levels)
{
    
    for(int level = 1; level < levels; level++) {

        for(int y = 0; y < editedImage[level].rows; y++) {
            for(int x = 0; x < editedImage[level].cols; x++) {
                        
                int kernelSize = 2;
                for(int py = 2 * y - kernelSize/2; py < 2 * y + kernelSize/2; py++) {
                    for(int px = 2 * x - kernelSize/2; px < 2 * x + kernelSize/2; px++) {

                        int pixel = py * editedImage[level - 1].cols + px;
                        if(pixel >= 0 && pixel < editedImage[level - 1].cols * editedImage[level - 1].rows) {
                            if(scribbleImage[level - 1].ptr<unsigned char>()[pixel] == 255) {
                                scribbleImage[level].ptr<unsigned char>()[y * scribbleImage[level].cols + x] = 255;
                                for(int ch = 0; ch < 3; ch++) {
                                    editedImage[level].ptr<unsigned char>()[(y * editedImage[level].cols + x) * 3 + ch] = editedImage[level - 1].ptr<unsigned char>()[pixel * 3 + ch];
                                }
                            }
                        }
                           
                    }
                }

            }
        }

    }
    
}

void paintImage(int x, int y, int scribbleColor, int scribbleRadius)
{

    for(int py = y - scribbleRadius/2; py < y + scribbleRadius/2; py++) {
        for(int px = x - scribbleRadius/2; px < x + scribbleRadius/2; px++) {

            int pixel = py * editedImage[0].cols + px;
            if(pixel >= 0 && pixel < editedImage[0].cols * editedImage[0].rows) {
                editedImage[0].ptr<unsigned char>()[pixel * 3 + 0] = scribbleColor;
                editedImage[0].ptr<unsigned char>()[pixel * 3 + 1] = scribbleColor;
                editedImage[0].ptr<unsigned char>()[pixel * 3 + 2] = scribbleColor;
                scribbleImage[0].ptr<unsigned char>()[pixel] = 255;
            }
        
        }
    }

}

void updateScribbleColor(int key) 
{

    if(key >= 48 && key <= 52) 
        scribbleColor = std::min((key - 48) * 64, 254);

}

void mouseEvent(int event, int x, int y, int flags, void *userData)
{

    if(event == cv::EVENT_LBUTTONDOWN)
        buttonIsPressed = true;

    if(event == cv::EVENT_LBUTTONUP)
        buttonIsPressed = false;

    if(event == cv::EVENT_MOUSEMOVE && buttonIsPressed && solverName.find("GPU") == std::string::npos)
        paintImage(x, y, scribbleColor, scribbleRadius);

#ifdef OPENCV_WITH_CUDA
	if (event == cv::EVENT_MOUSEMOVE && buttonIsPressed && solverName.find("GPU") != std::string::npos) {
		GPUPaintImage(x, y, scribbleColor, scribbleRadius, deviceEditedImage[0].ptr(), deviceEditedImage[0].step, deviceScribbleImage[0].ptr(),
			deviceScribbleImage[0].step, deviceScribbleImage[0].rows, deviceScribbleImage[0].cols);
		deviceEditedImage[0].download(editedImage[0]);
		deviceScribbleImage[0].download(scribbleImage[0]);
	}	
#endif
    
}

void trackbarEvent(int code, void* )
{

	switch (code) {
	case 0:
		solverName = "CPU-Jacobi";
		break;
	case 1:
		solverName = "CPU-Jacobi-Chebyshev";
		break;
	case 2:
		solverName = "CPU-GaussSeidel";
		break;
	case 3:
		solverName = "Eigen-BiCGStab";
		break;
	case 4:
		solverName = "Trilinos-AMG";
		break;
	case 5:
		solverName = "LAHBF";
		break;
	case 6:
		solverName = "GPU-Jacobi";
		break;
	case 7:
		solverName = "GPU-Jacobi-Chebyshev";
		break;
	case 8:
		solverName = "GPU-GaussSeidel";
		break;
	case 9:
		solverName = "CUSP-Jacobi";
		break;
	case 10:
		solverName = "CUSP-GaussSeidel";
		break;
	case 11:
		solverName = "CUSP-BiCGStab";
		break;
	case 12:
		solverName = "Paralution-BiCGStab";
		break;
	case 13:
		solverName = "ViennaCL-BiCGStab";
		break;
	}
	std::cout << solverName << " has been selected as solver" << std::endl;

}

int main(int argc, const char *argv[])
{

    if(argc == 1) {
        std::cout << "Usage: DepthDiffusion -i ImageFile.Extension" << std::endl;
        return 0;
    }

	//Read arguments
	std::string inputFileName, annotatedFileName;
	int solverCode = 7;
	std::string method = "Liao";
	bool live = false;
	bool artisticRendering = false;
	for(int argument = 1; argument < argc; argument++) {
		if (!strcmp(argv[argument], "-i"))
			inputFileName.assign(argv[argument + 1]);
		else if (!strcmp(argv[argument], "-a"))
			annotatedFileName.assign(argv[argument + 1]);
		else if (!strcmp(argv[argument], "--live"))
			live = true;
		else if (!strcmp(argv[argument], "-h"))
			std::cout << "Usage:\n -i input image\n -a annotated image\n";
	}

	//Initialize host variables
	cv::Mat originalImage = cv::imread(inputFileName);
	DepthEffect depthEffect(originalImage.rows, originalImage.cols);
	int pyrLevels = log2(std::max(std::min(originalImage.cols, originalImage.rows) / 45, 1)) + 1;
    std::vector<cv::Mat> depthImage;
    std::vector<cv::Mat> grayImage;
    editedImage.resize(pyrLevels);
    scribbleImage.resize(pyrLevels);
    depthImage.resize(pyrLevels);
    grayImage.resize(pyrLevels);
    for(int level = 0; level < pyrLevels; level++) {
        cv::Size pyrSize = cv::Size(originalImage.cols / powf(2, level), originalImage.rows / powf(2, level));
        editedImage[level] = cv::Mat::zeros(pyrSize, CV_8UC3);
		editedImage[level].setTo(cv::Scalar(0));
		scribbleImage[level] = cv::Mat::zeros(pyrSize, CV_8UC1);
		scribbleImage[level].setTo(cv::Scalar(0));
		depthImage[level] = cv::Mat::zeros(pyrSize, CV_8UC1);
		depthImage[level].setTo(cv::Scalar(255));
        grayImage[level] = cv::Mat::zeros(pyrSize, CV_8UC1);
		if (level == 0) cv::cvtColor(originalImage, grayImage[0], cv::COLOR_BGR2GRAY);
		else cv::pyrDown(grayImage[level - 1], grayImage[level]);
    }
#ifdef OPENCV_WITH_CUDA
	//Initialize device variables
	resetDevice();
	GpuMat deviceUCDepthImage = GpuMat(cv::Size(originalImage.cols, originalImage.rows), CV_8UC1);
	GpuMat deviceOriginalImage = GpuMat(originalImage);
	GpuMat deviceArtisticImage = GpuMat(cv::Size(originalImage.cols, originalImage.rows), CV_8UC3);
	std::vector<GpuMat> deviceGrayImage;
	std::vector<GpuMat> deviceDepthImage;
	std::vector<cv::Mat> floatDepthImage;
	deviceEditedImage.resize(pyrLevels);
	deviceScribbleImage.resize(pyrLevels);
	deviceDepthImage.resize(pyrLevels);
	deviceGrayImage.resize(pyrLevels);
	floatDepthImage.resize(pyrLevels);
	for (int level = 0; level < pyrLevels; level++) {
		cv::Size pyrSize = cv::Size(originalImage.cols / powf(2, level), originalImage.rows / powf(2, level));
		deviceEditedImage[level] = GpuMat(pyrSize, CV_8UC3); 
		deviceEditedImage[level].setTo(cv::Scalar(0));
		deviceScribbleImage[level] = GpuMat(pyrSize, CV_8UC1);
		deviceScribbleImage[level].setTo(cv::Scalar(0));
		deviceGrayImage[level] = GpuMat(pyrSize, CV_8UC1);
		deviceDepthImage[level] = GpuMat(pyrSize, CV_32FC1);
		deviceDepthImage[level].setTo(cv::Scalar(255));
		floatDepthImage[level] = cv::Mat(pyrSize, CV_32FC1);
		if (level == 0) cv::gpu::cvtColor(deviceOriginalImage, deviceGrayImage[0], cv::COLOR_BGR2GRAY);
		else {
		if (((deviceGrayImage[level - 1].rows + 1 / 2) == deviceGrayImage[level].rows) && ((deviceGrayImage[level - 1].cols + 1 / 2) == deviceGrayImage[level].cols))
				cv::gpu::pyrDown(deviceGrayImage[level - 1], deviceGrayImage[level]);
			else {
				deviceGrayImage[level - 1].download(grayImage[level - 1]);
				cv::pyrDown(grayImage[level - 1], grayImage[level]);
				deviceGrayImage[level].upload(grayImage[level]);
			}
		}
	}
	GPUAllocateDeviceMemory(originalImage.rows, originalImage.cols, pyrLevels);
#endif	
	//Initialize solver
	float beta = 0.4;
	float tolerance = 1e-4;
	int maxIterations = 1000;
	bool isDebugEnabled = false;
	scribbleRadius = std::min(originalImage.rows, originalImage.cols) * 0.02;
	Solver *solver = new Solver(originalImage.rows, originalImage.cols);
    solver->setBeta(beta);
    solver->setErrorThreshold(tolerance);
    solver->setMaximumNumberOfIterations(maxIterations);
	solver->setMaxLevel(pyrLevels - 1);
	if(isDebugEnabled) solver->enableDebug();
#ifdef OPENCV_WITH_CUDA
	GPULoadWeights(beta);
#endif
	//Load supplementary images (if any)
    editedImage[0] = cv::imread(inputFileName);
    if(annotatedFileName.size() > 1) {
        scribbleImage[0] = cv::imread(annotatedFileName, 0);
		for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
			if (scribbleImage[0].ptr<unsigned char>()[pixel] != 32) {
				for (int ch = 0; ch < 3; ch++)
					editedImage[0].ptr<unsigned char>()[pixel * 3 + ch] = scribbleImage[0].ptr<unsigned char>()[pixel];
				scribbleImage[0].ptr<unsigned char>()[pixel] = 255;
			}
		}

    } 
#ifdef OPENCV_WITH_CUDA
	deviceScribbleImage[0].upload(scribbleImage[0]);
	deviceEditedImage[0].upload(editedImage[0]);
#endif

    cv::namedWindow("Original Image");
    cv::namedWindow("Edited Image");
    cv::namedWindow("Depth Image");
	cv::setMouseCallback("Edited Image", mouseEvent, NULL);
	cv::createTrackbar("Solver", "Edited Image", &solverCode, 13, trackbarEvent);
	
    while(key != 27) {

        cv::imshow("Original Image", originalImage);
        cv::imshow("Edited Image", editedImage[0]);
        cv::imshow("Depth Image", depthImage[0]);
		if(artisticRendering) cv::imshow("Artistic Image", depthEffect.getArtisticImage());

        key = cv::waitKey(33);
        updateScribbleColor(key);
        
		//Check menu

		if (solverName.find("GPU") != std::string::npos) {
#ifdef OPENCV_WITH_CUDA
			if (key == 'b' || key == 'B') {
				std::cout << "Defocusing image..." << std::endl;
				GPUSimulateDefocus(deviceOriginalImage.ptr(), deviceOriginalImage.step, deviceDepthImage[0].ptr<float>(), deviceDepthImage[0].step,
					deviceArtisticImage.ptr(), deviceArtisticImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
				deviceArtisticImage.download(depthEffect.getArtisticImage());
				artisticRendering = true;
			}
			if (key == 'g' || key == 'G') {
				std::cout << "Desaturating image..." << std::endl;
				GPUSimulateDesaturation(deviceOriginalImage.ptr(), deviceOriginalImage.step, deviceGrayImage[0].ptr(),
					deviceGrayImage[0].step, deviceDepthImage[0].ptr<float>(), deviceDepthImage[0].step, deviceArtisticImage.ptr(),
					deviceArtisticImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
				deviceArtisticImage.download(depthEffect.getArtisticImage());
				artisticRendering = true;
			}
			if (key == 'h' || key == 'H') {
				std::cout << "Hazing image..." << std::endl;
				GPUSimulateHaze(deviceOriginalImage.ptr(), deviceOriginalImage.step, deviceDepthImage[0].ptr<float>(), deviceDepthImage[0].step,
					deviceArtisticImage.ptr(), deviceArtisticImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
				deviceArtisticImage.download(depthEffect.getArtisticImage());
				artisticRendering = true;
			}
#else
			std::cout << "GPU module not supported (OpenCV's GPU support is required)" << std::endl;
#endif
		} else {
			if (key == 'b' || key == 'B') {
				std::cout << "Defocusing image..." << std::endl;
				depthEffect.simulateDefocus(originalImage, depthImage[0]);
				artisticRendering = true;
			}
			if (key == 'g' || key == 'G') {
				std::cout << "Desaturating image..." << std::endl;
				depthEffect.simulateDesaturation(originalImage, grayImage[0], depthImage[0]);
				artisticRendering = true;
			}
			if (key == 'h' || key == 'H') {
				std::cout << "Hazing image..." << std::endl;
				depthEffect.simulateHaze(originalImage, depthImage[0]);
				artisticRendering = true;
			}
		}
		
        if(key == 'd' || key == 'D' || live) {
            
            double begin = cpuTime();
            
			solver->setMethod(method);
			if (solverName.find("GPU") != std::string::npos) {

#ifdef OPENCV_WITH_CUDA
				deviceScribbleImage[0].upload(scribbleImage[0]);
				deviceEditedImage[0].upload(editedImage[0]);
				for (int level = 1; level < pyrLevels; level++) {

					if(((deviceGrayImage[level - 1].rows + 1 / 2) == deviceGrayImage[level].rows) && ((deviceGrayImage[level - 1].cols + 1 / 2) == deviceGrayImage[level].cols))
						cv::gpu::pyrDown(deviceGrayImage[level - 1], deviceGrayImage[level]);
					else {
						deviceGrayImage[level - 1].download(grayImage[level - 1]);
						cv::pyrDown(grayImage[level - 1], grayImage[level]);
						deviceGrayImage[level].upload(grayImage[level]);
					}
					GPUPyrDownAnnotation(deviceScribbleImage[level - 1].ptr(), deviceScribbleImage[level - 1].step,
						deviceEditedImage[level - 1].ptr(), deviceEditedImage[level - 1].step, deviceEditedImage[level - 1].rows,
						deviceEditedImage[level - 1].cols, deviceScribbleImage[level].ptr(), deviceScribbleImage[level].step,
						deviceEditedImage[level].ptr(), deviceEditedImage[level].step, deviceEditedImage[level].rows,
						deviceEditedImage[level].cols);
				}

				GPUConvertToFloat(deviceEditedImage[pyrLevels - 1].ptr(), deviceEditedImage[pyrLevels - 1].step,
					deviceDepthImage[pyrLevels - 1].ptr<float>(), deviceDepthImage[pyrLevels - 1].step, deviceScribbleImage[pyrLevels - 1].ptr(), 
					deviceScribbleImage[pyrLevels - 1].step, deviceEditedImage[pyrLevels - 1].rows, deviceEditedImage[pyrLevels - 1].cols);
				
				for (int level = pyrLevels - 1; level >= 0; level--) {

					int CUDAIteration = maxIterations / powf(2.0, (pyrLevels - 1) - level);
					float CUDAThreshold = 1e-5;

					GPUMatrixFreeSolver(deviceDepthImage[level].ptr<float>(), deviceDepthImage[level].step, deviceScribbleImage[level].ptr(),
						deviceScribbleImage[level].step, deviceGrayImage[level].ptr(), deviceGrayImage[level].step, depthImage[level].rows,
						depthImage[level].cols, beta, CUDAIteration, CUDAThreshold, solverName, method, level, isDebugEnabled);
				
					if (level > 0) {
						
						if (deviceDepthImage[level].rows * 2 == deviceDepthImage[level - 1].rows && deviceDepthImage[level].cols * 2 == deviceDepthImage[level - 1].cols) {
							cv::gpu::pyrUp(deviceDepthImage[level], deviceDepthImage[level - 1]);
						} else {
							deviceDepthImage[level].download(floatDepthImage[level]);
							cv::pyrUp(floatDepthImage[level], floatDepthImage[level - 1], deviceDepthImage[level - 1].size());
							deviceDepthImage[level - 1].upload(floatDepthImage[level - 1]);
						}
						
					}

				}

				deviceDepthImage[0].convertTo(deviceUCDepthImage, CV_8UC1);
				deviceUCDepthImage.download(depthImage[0]);

#else
				std::cout << "GPU module not supported (OpenCV's GPU support is required)" << std::endl;
#endif
			} else {

				if (solverName == "Trilinos-AMG") {

					for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++)
						depthImage[0].ptr<unsigned char>()[pixel] = editedImage[0].ptr<unsigned char>()[pixel * 3 + 0];

					solver->runAMG(depthImage[0].ptr<unsigned char>(), scribbleImage[0].ptr<unsigned char>(), grayImage[0].ptr<unsigned char>(),
						depthImage[0].rows, depthImage[0].cols);

				}
				else if (solverName == "LAHBF") {

					for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++)
						depthImage[0].ptr<unsigned char>()[pixel] = editedImage[0].ptr<unsigned char>()[pixel * 3 + 0];

					//Currently, LAHBF does support edge-aware depth diffusion
					solver->runLAHBPCG(depthImage[0].ptr<unsigned char>(), scribbleImage[0].ptr<unsigned char>(), grayImage[0].ptr<unsigned char>(),
						depthImage[0].rows, depthImage[0].cols);

				}
				else {

					pyrDownAnnotation(editedImage, scribbleImage, pyrLevels);
	
					for (int pixel = 0; pixel < editedImage[pyrLevels - 1].rows * editedImage[pyrLevels - 1].cols; pixel++) {
						if(scribbleImage[pyrLevels - 1].ptr<unsigned char>()[pixel] == 255) depthImage[pyrLevels - 1].ptr<unsigned char>()[pixel] = editedImage[pyrLevels - 1].ptr<unsigned char>()[pixel * 3 + 0];
					}
						
					for (int level = pyrLevels - 1; level >= 0; level--) {

						if (method == "Macedo") {
							solver->setMaximumNumberOfIterations(1000 / powf(2.0, (pyrLevels - 1) - level));
							solver->setErrorThreshold(1e-5);
						}
						else {
							solver->setMaximumNumberOfIterations(maxIterations);
							solver->setErrorThreshold(1e-8);
						}

						solver->computePositions(depthImage[level].rows, depthImage[level].cols);
						solver->computeWeights(grayImage[level].ptr<unsigned char>(), depthImage[level].ptr<unsigned char>(), level, depthImage[level].rows, depthImage[level].cols);

						if (solverName.find("CPU") != std::string::npos)
							solver->runMatrixFreeSolver(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
								grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, solverName);
						else if (solverName == "Eigen-BiCGStab")
							solver->runConjugateGradient(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
								grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols);
						else if (solverName == "CUSP-Jacobi" || solverName == "CUSP-GaussSeidel")
							CUSPMatrixFreeSolver(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
								grayImage[level].ptr<unsigned char>(), solver->getWeights(), depthImage[level].rows, depthImage[level].cols, 
								beta, maxIterations, tolerance, solverName, isDebugEnabled);
						else if (solverName == "CUSP-BiCGStab")
							CUSPPCG(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
								grayImage[level].ptr<unsigned char>(), solver->getWeights(), depthImage[level].rows, depthImage[level].cols, 
								beta, maxIterations, tolerance, isDebugEnabled);
						else if (solverName == "Paralution-BiCGStab")
							ParalutionPCG(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
								grayImage[level].ptr<unsigned char>(), solver->getWeights(), depthImage[level].rows, depthImage[level].cols, 
								beta, maxIterations, tolerance);
						else if (solverName == "ViennaCL-BiCGStab")
							ViennaCLPCG(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
								grayImage[level].ptr<unsigned char>(), solver->getWeights(), depthImage[level].rows, depthImage[level].cols, 
								beta, maxIterations, tolerance);

						if (level > 0) cv::pyrUp(depthImage[level], depthImage[level - 1], depthImage[level - 1].size());

						
					}

				}
				
			}
			
			double end = cpuTime();
            std::cout << "Processing Time: " << (end - begin) * 1000 << " ms" << std::endl;
        
        }

		if(key == 's' || key == 'S') {

            cv::Mat imageToSave = cv::Mat::zeros(cv::Size(originalImage.cols, originalImage.rows), CV_8UC3);
            for(int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
                if(scribbleImage[0].ptr<unsigned char>()[pixel] != 255)
                    for(int ch = 0; ch < 3; ch++)
                        imageToSave.ptr<unsigned char>()[pixel * 3 + ch] = 32;
                else
                    for(int ch = 0; ch < 3; ch++)
                        imageToSave.ptr<unsigned char>()[pixel * 3 + ch] = editedImage[0].ptr<unsigned char>()[pixel * 3 + ch];
            }
            cv::imwrite("Scribble.png", imageToSave);
        
		}

		if (key == 't' || key == 'T') {

			cv::Mat imageToSave = cv::Mat::zeros(cv::Size(originalImage.cols, originalImage.rows), CV_8UC3);
			for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
				for (int ch = 0; ch < 3; ch++)
					imageToSave.ptr<unsigned char>()[pixel * 3 + ch] = depthImage[0].ptr<unsigned char>()[pixel];
			}
			cv::imwrite("A.png", imageToSave);

		}
		/*
		if (key == 'p' || key == 'P') {

			cv::imshow("P1", grayImage[0]);
			cv::imshow("P2", grayImage[1]);
			cv::imshow("P3", grayImage[2]);
			cv::imshow("P4", grayImage[3]);
			
			cv::imshow("D1", depthImage[0]);
			cv::imshow("D2", depthImage[1]);
			cv::imshow("D3", depthImage[2]);
			cv::imshow("D4", depthImage[3]);

		
		}
		*/

		if (key == 'm' || key == 'M') {

			method = (method == "Liao") ? "Macedo" : "Liao";
			std::cout << method << "'s method has been enabled" << std::endl;
		
		}
		
    }

    delete solver;
	GPUFreeDeviceMemory(pyrLevels);
    return 0;

}