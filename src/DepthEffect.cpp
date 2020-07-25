#include "DepthEffect.h"

DepthEffect::DepthEffect(int rows, int cols)
{
	artisticImage = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC3);
}

void DepthEffect::simulateDefocus(cv::Mat originalImage, cv::Mat depthImage)
{

	int kernelSize = 0.025 * sqrtf(powf(originalImage.rows, 2) + powf(originalImage.cols, 2));
	for (int pixel = 0; pixel < originalImage.rows * originalImage.cols; pixel++) {

		int anisotropicKernelSize = kernelSize * depthImage.ptr<unsigned char>()[pixel] / 255.0;
		int y = pixel / originalImage.cols;
		int x = pixel % originalImage.cols;
		float sum[3] = { 0 };
		float count = 0;

		for (int py = y - anisotropicKernelSize / 2; py < y + anisotropicKernelSize / 2; py++) {
			for (int px = x - anisotropicKernelSize / 2; px < x + anisotropicKernelSize / 2; px++) {

				int kernelPixel = py * originalImage.cols + px;
				if (px >= 0 && py >= 0 && px < originalImage.cols && py < originalImage.rows) {

					for (int channel = 0; channel < originalImage.channels(); channel++)
						sum[channel] += originalImage.ptr<unsigned char>()[kernelPixel * 3 + channel];
					count++;

				}
			}
		}

		if (count == 0) {
			for (int channel = 0; channel < originalImage.channels(); channel++)
				artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = originalImage.ptr<unsigned char>()[pixel * 3 + channel];
		}
		else {
			for (int channel = 0; channel < originalImage.channels(); channel++)
				artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = sum[channel] / count;
		}

	}

}

void DepthEffect::simulateDesaturation(cv::Mat originalImage, cv::Mat grayImage, cv::Mat depthImage)
{

	for (int pixel = 0; pixel < originalImage.rows * originalImage.cols; pixel++) {

		float f = depthImage.ptr<unsigned char>()[pixel] / 255.0;
		for (int channel = 0; channel < originalImage.channels(); channel++) {
			artisticImage.ptr<unsigned char>()[pixel * 3 + channel] =
				(1.0 - f) * originalImage.ptr<unsigned char>()[pixel * 3 + channel] + f * grayImage.ptr<unsigned char>()[pixel];
		}
	}

}

void DepthEffect::simulateHaze(cv::Mat originalImage, cv::Mat depthImage)
{

	float beta = 2;
	for (int pixel = 0; pixel < originalImage.rows * originalImage.cols; pixel++) {

		float t = expf(-beta * depthImage.ptr<unsigned char>()[pixel] / 255.0);
		for (int channel = 0; channel < originalImage.channels(); channel++) {
			artisticImage.ptr<unsigned char>()[pixel * 3 + channel] = t * originalImage.ptr<unsigned char>()[pixel * 3 + channel] + (1 - t) * 255;
		}

	}

}