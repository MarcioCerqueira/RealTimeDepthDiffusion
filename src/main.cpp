/*
References:
	Diffusion Curves - "Diffusion Curves: A Vector Representation for Smooth-Shaded Images" by Orzan et al. ToG 2008
	Haze Simulation - Equations (1, 2) of "Single Image Haze Removal using Dark Channel Prior" by He et al. CVPR 2009
    Chebyshev Semi-Iterative Method - "A Chebyshev Semi-Iterative Approach for Accelerating Projective and PBD" by H. Wang ToG 2015 
    Depth Design - "Depth Annotations: Designing Depth of a Single Image for Depth-based Effects" by Liao et al. GI 2017
*/

#include "GPUSolver.h"
#include "GPUDepthEffect.h"
#include "GPUImageProcessing.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv::cuda;

bool buttonIsPressed = false;
int key = 0;
int scribbleColor = 0;
int scribbleRadius;
std::vector<cv::Mat> editedImage;
std::vector<cv::Mat> scribbleImage;
std::vector<GpuMat> deviceEditedImage;
std::vector<GpuMat> deviceScribbleImage;

double cpuTime(void)
{

	double value;
	value = (double) clock () / (double) CLOCKS_PER_SEC;
	return value;

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

	if (event == cv::EVENT_MOUSEMOVE && buttonIsPressed) {
		GPUPaintImage(x, y, scribbleColor, scribbleRadius, deviceEditedImage[0].ptr(), deviceEditedImage[0].step, deviceScribbleImage[0].ptr(),
			deviceScribbleImage[0].step, deviceScribbleImage[0].rows, deviceScribbleImage[0].cols);
		deviceEditedImage[0].download(editedImage[0]);
		deviceScribbleImage[0].download(scribbleImage[0]);
	}	
    
}

int main(int argc, const char *argv[])
{

	if (argc == 1) {
		std::cout << "Usage: DepthDiffusion -i ImageFile.Extension" << std::endl;
		return 0;
	}

	std::string inputFileName, annotatedFileName;
	bool live = false;
	bool artisticRendering = false;
	bool hazeEffect = false;
	bool desaturationEffect = false;
	bool refocusEffect = false;
	double begin, end;

	//Read arguments
	for (int argument = 1; argument < argc; argument++) {
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
	cv::Mat artisticImage = cv::Mat::zeros(originalImage.rows, originalImage.cols, CV_8UC3);
	int pyrLevels = log2(std::max(std::min(originalImage.cols, originalImage.rows) / 45, 1)) + 1;
	std::vector<cv::Mat> depthImage;
	std::vector<cv::Mat> grayImage;
	editedImage.resize(pyrLevels);
	scribbleImage.resize(pyrLevels);
	depthImage.resize(pyrLevels);
	grayImage.resize(pyrLevels);
	for (int level = 0; level < pyrLevels; level++) {
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
		if (level == 0) cv::cuda::cvtColor(deviceOriginalImage, deviceGrayImage[0], cv::COLOR_BGR2GRAY);
		else {
			if (((deviceGrayImage[level - 1].rows + 1 / 2) == deviceGrayImage[level].rows) && ((deviceGrayImage[level - 1].cols + 1 / 2) == deviceGrayImage[level].cols))
				cv::cuda::pyrDown(deviceGrayImage[level - 1], deviceGrayImage[level]);
			else {
				deviceGrayImage[level - 1].download(grayImage[level - 1]);
				cv::pyrDown(grayImage[level - 1], grayImage[level]);
				deviceGrayImage[level].upload(grayImage[level]);
			}
		}
	}
	GPUAllocateDeviceMemory(originalImage.rows, originalImage.cols, pyrLevels);
	
	//Initialize solver
	float beta = 0.4;
	int maxIterations = 1000;
	scribbleRadius = std::min(originalImage.rows, originalImage.cols) * 0.02;
	GPULoadWeights(beta);

	//Load supplementary images (if any)
	editedImage[0] = cv::imread(inputFileName);

	if (annotatedFileName.size() > 1) {
        scribbleImage[0] = cv::imread(annotatedFileName, 0);
		for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
			if (scribbleImage[0].ptr<unsigned char>()[pixel] != 32) {
				for (int ch = 0; ch < 3; ch++)
					editedImage[0].ptr<unsigned char>()[pixel * 3 + ch] = scribbleImage[0].ptr<unsigned char>()[pixel];
				scribbleImage[0].ptr<unsigned char>()[pixel] = 255;
			}
		}

    } 

	deviceScribbleImage[0].upload(scribbleImage[0]);
	deviceEditedImage[0].upload(editedImage[0]);
	
    cv::namedWindow("Original Image");
    cv::namedWindow("Edited Image");
    cv::namedWindow("Depth Image");
	cv::setMouseCallback("Edited Image", mouseEvent, NULL);

    while(key != 27) {

        cv::imshow("Original Image", originalImage);
        cv::imshow("Edited Image", editedImage[0]);
		cv::imshow("Depth Image", depthImage[0]);
		if(artisticRendering) cv::imshow("Artistic Image", artisticImage);

        key = cv::waitKey(33);
        updateScribbleColor(key);
		
		if (key == 'b' || key == 'B' || refocusEffect) {

			GPUSimulateDefocus(deviceOriginalImage.ptr(), deviceOriginalImage.step, deviceDepthImage[0].ptr<float>(), 
				deviceDepthImage[0].step, deviceArtisticImage.ptr(), deviceArtisticImage.step, 
				deviceOriginalImage.rows, deviceOriginalImage.cols);
			deviceArtisticImage.download(artisticImage);
			
			artisticRendering = true;
			refocusEffect = true;
			desaturationEffect = false;
			hazeEffect = false;
		
		}

		if (key == 'g' || key == 'G' || desaturationEffect) {
		
			GPUSimulateDesaturation(deviceOriginalImage.ptr(), deviceOriginalImage.step, deviceGrayImage[0].ptr(),
				deviceGrayImage[0].step, deviceDepthImage[0].ptr<float>(), deviceDepthImage[0].step, 
				deviceArtisticImage.ptr(), 	deviceArtisticImage.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
			deviceArtisticImage.download(artisticImage);
		
			artisticRendering = true;
			desaturationEffect = true;
			refocusEffect = false;
			hazeEffect = false;
		
		}

		if (key == 'h' || key == 'H' || hazeEffect) {

			GPUSimulateHaze(deviceOriginalImage.ptr(), deviceOriginalImage.step, deviceDepthImage[0].ptr<float>(), 
				deviceDepthImage[0].step, deviceArtisticImage.ptr(), deviceArtisticImage.step, 
				deviceOriginalImage.rows, deviceOriginalImage.cols);
			deviceArtisticImage.download(artisticImage);

			artisticRendering = true;
			hazeEffect = true;
			desaturationEffect = false;
			refocusEffect = false;
		
		}
		
        if(key == 'd' || key == 'D' || live) {
            
            begin = cpuTime();
				
			deviceScribbleImage[0].upload(scribbleImage[0]);
			deviceEditedImage[0].upload(editedImage[0]);
			
			for (int level = 1; level < pyrLevels; level++) {

				if (((deviceGrayImage[level - 1].rows + 1 / 2) == deviceGrayImage[level].rows) && ((deviceGrayImage[level - 1].cols + 1 / 2) == deviceGrayImage[level].cols))
					cv::cuda::pyrDown(deviceGrayImage[level - 1], deviceGrayImage[level]);
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
					depthImage[level].cols, beta, CUDAIteration, CUDAThreshold, level);
					
				if (level > 0) {

					if (deviceDepthImage[level].rows * 2 == deviceDepthImage[level - 1].rows && deviceDepthImage[level].cols * 2 == deviceDepthImage[level - 1].cols) {
						cv::cuda::pyrUp(deviceDepthImage[level], deviceDepthImage[level - 1]);
					}
					else {
						deviceDepthImage[level].download(floatDepthImage[level]);
						cv::pyrUp(floatDepthImage[level], floatDepthImage[level - 1], deviceDepthImage[level - 1].size());
						deviceDepthImage[level - 1].upload(floatDepthImage[level - 1]);
					}

					GPUConvertToFloat(deviceEditedImage[level - 1].ptr(), deviceEditedImage[level - 1].step,
						deviceDepthImage[level - 1].ptr<float>(), deviceDepthImage[level - 1].step, deviceScribbleImage[level - 1].ptr(),
						deviceScribbleImage[level - 1].step, deviceEditedImage[level - 1].rows, deviceEditedImage[level - 1].cols);

				}
					

			}
				
			deviceDepthImage[0].convertTo(deviceUCDepthImage, CV_8UC1);
			deviceUCDepthImage.download(depthImage[0]);
				
			end = cpuTime();
            
        }

		if(key == 's' || key == 'S') {

            cv::Mat imageToSave = cv::Mat::zeros(cv::Size(originalImage.cols, originalImage.rows), CV_8UC3);
			for(int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
                for(int ch = 0; ch < 3; ch++)
					imageToSave.ptr<unsigned char>()[pixel * 3 + ch] = editedImage[0].ptr<unsigned char>()[pixel * 3 + ch];
            }
            cv::imwrite("AnnotatedImage.png", imageToSave);
        
			for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
				for (int ch = 0; ch < 3; ch++)
					imageToSave.ptr<unsigned char>()[pixel * 3 + ch] = depthImage[0].ptr<unsigned char>()[pixel];
			}
			cv::imwrite("DepthMap.png", imageToSave);

			for (int pixel = 0; pixel < editedImage[0].rows * editedImage[0].cols; pixel++) {
				for (int ch = 0; ch < 3; ch++)
					imageToSave.ptr<unsigned char>()[pixel * 3 + ch] = artisticImage.ptr<unsigned char>()[pixel * 3 + ch];
			}
			cv::imwrite("ArtisticEffect.png", imageToSave);
			std::cout << "Saving images..." << std::endl;
		}

		if (key == 't' || key == 'T') {
			std::cout << "Processing Time: " << (end - begin) * 1000 << " ms" << std::endl;
		}

		if (key == '-') {
			scribbleRadius -= 2;
			std::cout << "Scribble Radius: " << scribbleRadius << std::endl;
		}

		if (key == '+') {
			scribbleRadius += 2;
			std::cout << "Scribble Radius: " << scribbleRadius << std::endl;
		}
		
    }

	GPUFreeDeviceMemory(pyrLevels);
    return 0;

}