#ifndef DEPTH_EFFECT_H
#define DEPTH_EFFECT_H

#include <opencv2/opencv.hpp>

class DepthEffect
{
public:
	DepthEffect(int rows, int cols);
	void simulateDefocus(cv::Mat originalImage, cv::Mat depthImage);
	void simulateDesaturation(cv::Mat originalImage, cv::Mat grayImage, cv::Mat depthImage);
	void simulateHaze(cv::Mat originalImage, cv::Mat depthImage);
	cv::Mat& getArtisticImage() { return artisticImage; }
private:
	cv::Mat artisticImage;
};
#endif