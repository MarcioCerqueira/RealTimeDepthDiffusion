To compile this project, you will need to have the following libraries:
	
	OpenCV (with GPU support) - To load and process the images;

	Eigen - To run the conjugate gradient solver (Solver->runConjugateGradient);

	Trilinos - To run the algebraic multigrid solver (Solver->runAMG);

	CUDA - To run the GPU solvers;

	(Optional) CUSP - To run GPU-based Jacobi, Gauss-Seidel and BiCGStab solvers;

	(Optional) Paralution - To run Paralution's BiCGStab solver;

	(Optional) ViennaCL - To run ViennaCL's BiCGStab solver;

To disable the optional libraries from compilation, comment the "#define" lines in GPUSolver.cu

The software accepts as arguments:

	-i: Input image

	-a: Annnotated image

Menu:

	The solver trackbar enables runtime selection of a CPU-based or GPU-based solver

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
	
	Press 'm' or 'M' to change between our approach and related work