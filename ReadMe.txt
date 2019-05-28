To compile this project, you will need to have the following libraries:
	
	OpenCV - To load and process the images;

	Eigen - To run the conjugate gradient solver (Solver->runConjugateGradient);

	Trilinos - To run the algebraic multigrid solver (Solver->runAMG);

The software accepts as arguments an image and optionally the corresponding annotated image

Times measured over a 800 x 900 image on an i5:

	CPU-Based Pyramidal Gauss-Seidel: 9 seconds
	CPU-Based AMG Conjugate Gradient: 4 seconds
	CPU-Based Pyramidal Conjugate Gradient: 2 seconds

Menu:

	Press '0' to annotate the "Edited Image" with depth 0

	Press '1' to annotate the "Edited Image" with depth 64

	Press '2' to annotate the "Edited Image" with depth 128

	Press '3' to annotate the "Edited Image" with depth 192

	Press '4' to annotate the "Edited Image" with depth 255

	Press 'd' or 'D' to start the depth diffusion process:
		The resulting depth map will be shown in the "Depth Image"

	Press 's' or 'S' to save the user-assited depth annotation;

	Press 'g' or 'G' to desaturate the original image:
		This effect must be activated after the depth diffusion process
		The resulting image will be shown in the "Artistic Image"

	Press 'h' or 'H' to add haze in the original image:
		This effect must be activated after the depth diffusion process
		The resulting image will be shown in the "Artistic Image"

	Press 'b' or 'B' to add defocus in the original image:
		This effect must be activated after the depth diffusion process
		The resulting image will be shown in the "Artistic Image"
	