To compile this project, you will need to have both OpenCV and Eigen libraries installed.

I use the following command to compile the project: "g++ main.cpp Solver.cpp -O3 -march=native -IC:/Eigen/include -IC:/OpenCV/opencv/build/include -LC:/OpenCV/opencv/build/x86/mingw/bin -llibopencv_core231 -llibopencv_imgproc231 -llibopencv_highgui231 -o main.exe images/LowHorse.jpg"

Menu:

	Press '0' to annotate the "Edited Image" with depth 0
	Press '1' to annotate the "Edited Image" with depth 64
	Press '2' to annotate the "Edited Image" with depth 128
	Press '3' to annotate the "Edited Image" with depth 192
	Press '4' to annotate the "Edited Image" with depth 255
	Press 'd' or 'D' to start the depth diffusion process:
		The resulting depth map will be shown in the "Depth Image"
	Press 's' or 'S' to desaturate the original image:
		This effect must be activated after the depth diffusion process
		The resulting image will be shown in the "Artistic Image"
	Press 'h' or 'H' to add haze in the original image:
		This effect must be activated after the depth diffusion process
		The resulting image will be shown in the "Artistic Image"
	Press 'b' or 'B' to add defocus in the original image:
		This effect must be activated after the depth diffusion process
		The resulting image will be shown in the "Artistic Image"
	