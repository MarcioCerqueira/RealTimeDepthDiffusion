#ifndef GPU_IMAGE_PROCESSING_H
#define GPU_IMAGE_PROCESSING_H

void GPUConvertToFloat(unsigned char *src, size_t srcPitch, float *dst, size_t dstPitch, unsigned char *mask, size_t maskPitch,
	int rows, int cols);
void GPUPyrDownAnnotation(unsigned char *prevScribbleImage, size_t prevScribblePitch, unsigned char *prevEditedImage,
	size_t prevEditedPitch, int previousRows, int previousCols, unsigned char *currScribbleImage, size_t currScribblePitch,
	unsigned char *currEditedImage, size_t currEditedPitch, int currentRows, int currentCols);
void GPUPaintImage(int x, int y, int scribbleColor, int scribbleRadius, unsigned char *editedImage, size_t editedPitch, 
	unsigned char *scribbleImage, size_t scribblePitch, int rows, int cols);

#endif