/*
References:
    Haze Simulation - Equations (1, 2) of "Single Image Haze Removal using Dark Channel Prior" by He et al. CVPR 2009
    Diffusion Curves - "Diffusion Curves: A Vector Representation for Smooth-Shaded Images" by Orzan et al. ToG 2008
    Depth Design - "Depth Annotations: Designing Depth of a Single Image for Depth-based Effects" by Liao et al. GI 2017
*/

#include "Solver.h"
#include <vector>
#include <opencv2/opencv.hpp>

bool buttonIsPressed = false;
int key = 0;
int scribbleColor = 0;
int scribbleRadius;
std::vector<cv::Mat> editedImage;
std::vector<cv::Mat> scribbleImage;

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

int main(int argc, char **argv)
{

    if(argc == 1) {
        std::cout << "Usage: DepthDiffusion.exe ImageFile.Extension" << std::endl;
        return 0;
    }

    cv::Mat originalImage = cv::imread(argv[1]);

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
    cv::Mat artisticImage = cv::Mat::zeros(cv::Size(originalImage.cols, originalImage.rows), CV_8UC3);
    
    Solver *solver = new Solver(originalImage.rows, originalImage.cols);
    solver->setBeta(0.4);
    solver->setErrorThreshold(1e-4);
    solver->setMaximumNumberOfIterations(1000);
    //solver->enableDebug();
    
    editedImage[0] = cv::imread(argv[1]);
    if(argc == 3) {
        scribbleImage[0] = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
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
       
    while(key != 27) {

        cv::imshow("Original Image", originalImage);
        cv::imshow("Edited Image", editedImage[0]);
        cv::imshow("Depth Image", depthImage[0]);
        cv::imshow("Artistic Image", artisticImage);

        key = cv::waitKey(33);
        updateScribbleColor(key);
        
        //if key == 'd' or key == 'D'
        if(key == 68 || key == 100) {
            
            double begin = cpuTime();
            pyrDownAnnotation(editedImage, scribbleImage, pyrLevels);
            
			for(int pixel = 0; pixel < editedImage[pyrLevels - 1].rows * editedImage[pyrLevels - 1].cols; pixel++)
                depthImage[pyrLevels - 1].ptr<unsigned char>()[pixel] = editedImage[pyrLevels - 1].ptr<unsigned char>()[pixel*3+0];

			for(int level = pyrLevels - 1; level >= 0; level--) {

				solver->runConjugateGradient(depthImage[level].ptr<unsigned char>(), scribbleImage[level].ptr<unsigned char>(),
					grayImage[level].ptr<unsigned char>(), depthImage[level].rows, depthImage[level].cols);
				if(level > 0) {
					cv::Size pyrSize = cv::Size(depthImage[level - 1].cols, depthImage[level - 1].rows);
					cv::pyrUp(depthImage[level], depthImage[level - 1], pyrSize);
				}
			
			}
			
			/*
            // For AMG
            for(int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++)
                depthImage[0].ptr<unsigned char>()[pixel] = editedImage[0].ptr<unsigned char>()[pixel*3+0];
            
			solver->runAMG(depthImage[0].ptr<unsigned char>(), scribbleImage[0].ptr<unsigned char>(), grayImage[0].ptr<unsigned char>(), 
				depthImage[0].rows, depthImage[0].cols);
			*/

            double end = cpuTime();
            std::cout << "CPU: " << (end - begin) * 1000 << " ms" << std::endl;
        
        }

        //if key == 'g' or key == 'G'
        if(key == 71 || key == 103)
            desaturateImage(originalImage, grayImage[0], depthImage[0], artisticImage);

        //if key == 'h' or key == 'H'
        if(key == 72 || key == 104)
            hazeImage(originalImage, depthImage[0], artisticImage);

        //if key == 'b' or key == 'B'
        if(key == 66 || key == 98)
            defocusImage(originalImage, depthImage[0], artisticImage);
    
        //if key == 's' or key == 'S'
        if(key == 83 || key == 115) {
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

    delete [] solver;
    return 0;

}