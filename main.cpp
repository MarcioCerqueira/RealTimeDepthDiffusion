/*
References:
    Haze Simulation - Equations (1, 2) of "Single Image Haze Removal using Dark Channel Prior" by He et al. CVPR 2009
    Diffusion Curves - "Diffusion Curves: A Vector Representation for Smooth-Shaded Images" by Orzan et al. ToG 2008
    Depth Design - "Depth Annotations: Designing Depth of a Single Image for Depth-based Effects" by Liao et al. GI 2017
	Locally Adapted Hierarchical Basis Preconditioning by R. Szeliski ToG 2006
		Source code was directly adapted from "https://github.com/s-gupta/rgbdutils/blob/master/imagestack/src/LAHBPCG.cpp"
*/

#include "Solver.h"
#include "GPUSolver.h"
#include <vector>
#include <opencv2/opencv.hpp>

bool buttonIsPressed = false;
int key = 0;
int scribbleColor = 0;
int scribbleRadius;
std::vector<cv::Mat> editedImage;
std::vector<cv::Mat> scribbleImage;
std::string solverName = "Eigen-BiCGStab";

double cpuTime(void)
{

	double value;
	value = (double) clock () / (double) CLOCKS_PER_SEC;
	return value;

}

void desaturateImage(cv::Mat originalImage, cv::Mat grayImage, cv::Mat depthImage, cv::Mat &artisticImage)
{

    for(int pixel = 0; pixel < originalImage.rows * originalImage.cols; pixel++) {
        
        float f = depthImage.ptr<unsigned char>()[pixel]/255.0;
        for(int channel = 0; channel < originalImage.channels(); channel++) {
            artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = 
                (1.0 - f) * originalImage.ptr<unsigned char>()[pixel * 3 + channel] + f * grayImage.ptr<unsigned char>()[pixel];
        }
    }

}

void hazeImage(cv::Mat originalImage, cv::Mat depthImage, cv::Mat &artisticImage)
{

    float beta = 2;
    for(int pixel = 0; pixel < originalImage.rows * originalImage.cols; pixel++) {
        
        float t = expf(-beta * depthImage.ptr<unsigned char>()[pixel]/255.0);
        for(int channel = 0; channel < originalImage.channels(); channel++) {
            artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = t * originalImage.ptr<unsigned char>()[pixel * 3 + channel] + (1 - t) * 255;
        }

    }

}

void defocusImage(cv::Mat originalImage, cv::Mat depthImage, cv::Mat &artisticImage) 
{

    int kernelSize = 0.025 * sqrtf(powf(originalImage.rows, 2) + powf(originalImage.cols, 2));
    for(int pixel = 0; pixel < originalImage.rows * originalImage.cols; pixel++) {

        int anisotropicKernelSize = kernelSize * depthImage.ptr<unsigned char>()[pixel]/255.0;
        int y = pixel / originalImage.cols;
        int x = pixel % originalImage.cols;
        float sum[3] = {0};
        float count = 0;
        
        for(int py = y - anisotropicKernelSize/2; py < y + anisotropicKernelSize/2; py++) {
            for(int px = x - anisotropicKernelSize/2; px < x + anisotropicKernelSize/2; px++) {

                int kernelPixel = py * originalImage.cols + px;
                if(kernelPixel >= 0 && kernelPixel < originalImage.cols * originalImage.rows) {

                    for(int channel = 0; channel < originalImage.channels(); channel++)
                        sum[channel] += originalImage.ptr<unsigned char>()[kernelPixel * 3 + channel];
                    count++;

                }
            }
        }
        
        if(count == 0) {
            for(int channel = 0; channel < originalImage.channels(); channel++)
                artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = originalImage.ptr<unsigned char>()[pixel * 3 + channel];
        } else {
            for(int channel = 0; channel < originalImage.channels(); channel++)
                artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = sum[channel] / count;
        }

    }
    
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

    if(event == cv::EVENT_MOUSEMOVE && buttonIsPressed)
        paintImage(x, y, scribbleColor, scribbleRadius);
    
}

void trackbarEvent(int code, void* )
{

	switch (code) {
	case 0:
		solverName = "CPU-Jacobi";
		break;
	case 1:
		solverName = "CPU-GaussSeidel";
		break;
	case 2:
		solverName = "Eigen-BiCGStab";
		break;
	case 3:
		solverName = "Trilinos-AMG";
		break;
	case 4:
		solverName = "LAHBF";
		break;
	case 5:
		solverName = "GPU-Jacobi";
		break;
	case 6:
		solverName = "GPU-GaussSeidel";
		break;
	case 7:
		solverName = "CUSP-Jacobi";
		break;
	case 8:
		solverName = "CUSP-GaussSeidel";
		break;
	case 9:
		solverName = "CUSP-BiCGStab";
		break;
	case 10:
		solverName = "Paralution-BiCGStab";
		break;
	case 11:
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
	int solverCode = 2;
	for(int argument = 1; argument < argc - 1; argument++) {
		if (!strcmp(argv[argument], "-i"))
			inputFileName.assign(argv[argument + 1]);
		else if (!strcmp(argv[argument], "-a"))
			annotatedFileName.assign(argv[argument + 1]);
		else if (!strcmp(argv[argument], "-h"))
			std::cout << "Usage:\n -i input image\n -a annotated image\n";
	}

	//Initialize variables
    cv::Mat originalImage = cv::imread(inputFileName);
	cv::Mat artisticImage = cv::Mat::zeros(cv::Size(originalImage.cols, originalImage.rows), CV_8UC3);
	int pyrLevels = log2(std::max(std::min(originalImage.cols, originalImage.rows) / 64, 1)) + 1;
    std::vector<cv::Mat> depthImage;
    std::vector<cv::Mat> grayImage;
    editedImage.resize(pyrLevels);
    scribbleImage.resize(pyrLevels);
    depthImage.resize(pyrLevels);
    grayImage.resize(pyrLevels);
    for(int level = 0; level < pyrLevels; level++) {
        cv::Size pyrSize = cv::Size(originalImage.cols / powf(2, level), originalImage.rows / powf(2, level));
        editedImage[level] = cv::Mat::zeros(pyrSize, CV_8UC3);
        scribbleImage[level] = cv::Mat::zeros(pyrSize, CV_8UC1);
        depthImage[level] = cv::Mat::zeros(pyrSize, CV_8UC1);
        grayImage[level] = cv::Mat::zeros(pyrSize, CV_8UC1);
    }
    
	float beta = 0.4;
	float tolerance = 1e-4;
	int maxIterations = 1000;
	bool isDebugEnabled = false;
    Solver *solver = new Solver(originalImage.rows, originalImage.cols);
    solver->setBeta(beta);
    solver->setErrorThreshold(tolerance);
    solver->setMaximumNumberOfIterations(maxIterations);
	if(isDebugEnabled) solver->enableDebug();
    
    editedImage[0] = cv::imread(inputFileName);
    if(annotatedFileName.size() > 1) {
        scribbleImage[0] = cv::imread(annotatedFileName, CV_LOAD_IMAGE_GRAYSCALE);
        for(int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
            if(scribbleImage[0].ptr<unsigned char>()[pixel] != 32) {
                for(int ch = 0; ch < 3; ch++)
                    editedImage[0].ptr<unsigned char>()[pixel * 3 + ch] = scribbleImage[0].ptr<unsigned char>()[pixel];
                scribbleImage[0].ptr<unsigned char>()[pixel] = 255;
            }
        }
    } 

    scribbleRadius = std::min(originalImage.rows, originalImage.cols) * 0.02;
    cv::cvtColor(originalImage, grayImage[0], cv::COLOR_BGR2GRAY);
    for(int level = 1; level < pyrLevels; level++)
        cv::pyrDown(grayImage[level - 1], grayImage[level], cv::Size(grayImage[level].cols, grayImage[level].rows));

    cv::namedWindow("Original Image");
    cv::namedWindow("Edited Image");
    cv::namedWindow("Depth Image");
    cv::namedWindow("Artistic Image");
	cv::setMouseCallback("Edited Image", mouseEvent, NULL);
	cv::createTrackbar("Solver", "Edited Image", &solverCode, 11, trackbarEvent);

    while(key != 27) {

        cv::imshow("Original Image", originalImage);
        cv::imshow("Edited Image", editedImage[0]);
        cv::imshow("Depth Image", depthImage[0]);
        cv::imshow("Artistic Image", artisticImage);

        key = cv::waitKey(33);
        updateScribbleColor(key);
        
		//Check menu
		if (key == 'g' || key == 'G') {
			std::cout << "Desaturating image..." << std::endl;
			desaturateImage(originalImage, grayImage[0], depthImage[0], artisticImage);
		}
		if (key == 'h' || key == 'H') {
			std::cout << "Hazing image..." << std::endl;
			hazeImage(originalImage, depthImage[0], artisticImage);
		}
		if (key == 'b' || key == 'B') {
			std::cout << "Defocusing image..." << std::endl;
			defocusImage(originalImage, depthImage[0], artisticImage);
		}
		
        if(key == 'd' || key == 'D') {
            
            double begin = cpuTime();
            
			if (solverName == "Trilinos-AMG") {

				for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++)
					depthImage[0].ptr<unsigned char>()[pixel] = editedImage[0].ptr<unsigned char>()[pixel * 3 + 0];

				solver->runAMG(depthImage[0].ptr<unsigned char>(), scribbleImage[0].ptr<unsigned char>(), grayImage[0].ptr<unsigned char>(),
					depthImage[0].rows, depthImage[0].cols);

			} else if (solverName == "LAHBF") {

				for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++)
					depthImage[0].ptr<unsigned char>()[pixel] = editedImage[0].ptr<unsigned char>()[pixel * 3 + 0];

				//Currently, LAHBF does support edge-aware depth diffusion
				solver->runLAHBPCG(depthImage[0].ptr<unsigned char>(), scribbleImage[0].ptr<unsigned char>(), grayImage[0].ptr<unsigned char>(),
					depthImage[0].rows, depthImage[0].cols);

			} else {

				pyrDownAnnotation(editedImage, scribbleImage, pyrLevels);

				for (int pixel = 0; pixel < editedImage[pyrLevels - 1].rows * editedImage[pyrLevels - 1].cols; pixel++)
					depthImage[pyrLevels - 1].ptr<unsigned char>()[pixel] = editedImage[pyrLevels - 1].ptr<unsigned char>()[pixel * 3 + 0];
				
				for (int level = pyrLevels - 1; level >= 0; level--) {

					if (solverName == "CPU-Jacobi")
						solver->runJacobi(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols);
					else if(solverName == "CPU-GaussSeidel")
						solver->runGaussSeidel(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols);
					else if(solverName == "Eigen-BiCGStab")
						solver->runConjugateGradient(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols);
					else if(solverName == "GPU-Jacobi")
						GPUJacobi(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance);
					else if(solverName == "GPU-GaussSeidel")
						GPUGaussSeidel(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance);
					else if(solverName == "CUSP-Jacobi")
						CUSPJacobi(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance);
					else if (solverName == "CUSP-GaussSeidel")
						CUSPGaussSeidel(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance);
					else if(solverName == "CUSP-BiCGStab")
						CUSPPCG(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance);
					else if(solverName == "Paralution-BiCGStab")
						ParalutionPCG(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance);
					else if(solverName == "ViennaCL-BiCGStab")
						ViennaCLPCG(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
							grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols, beta, maxIterations,
							tolerance); 

					if (level > 0) {
						cv::Size pyrSize = cv::Size(depthImage[level - 1].cols, depthImage[level - 1].rows);
						cv::pyrUp(depthImage[level], depthImage[level - 1], pyrSize);
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
        
    }

    delete solver;
    return 0;

}