#include "Solver.h"

Solver::Solver(int rows, int cols)
{
    
	this->rows = rows;
	this->cols = cols;
	this->isDebugEnabled = false;
	
	image = (float*)malloc(sizeof(float) * rows * cols);
    debugImage = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1); 
	weights = (float*)malloc(rows * cols * 4 * sizeof(float));
	positions = (int*)malloc(rows * cols * 4 * sizeof(int));
	edges = (int*)malloc(rows * cols * sizeof(int));
	
}

Solver::~Solver()
{

	delete [] image;
	delete [] weights;
	delete [] positions;
	delete [] edges;

}

void Solver::runMatrixFreeSolver(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	std::string solverMethod)
{

	float error = 0;
	int iteration;
	float *previousImage = (float*)malloc(sizeof(float) * rows * cols);
	float *nextImage = (float*)malloc(sizeof(float) * rows * cols);
	
    //Parameters for Jacobi + Chebyshev
	int S = 10;
	float omega = 1.9;
	float rho = 0.99;
	float gamma = 0.99;
	//Parameter for Gauss-Seidel
	float theta = 1.9;

	for (int pixel = 0; pixel < rows * cols; pixel++) {
		image[pixel] = depthImage[pixel];
		if (solverMethod != "CPU-GaussSeidel") {
			previousImage[pixel] = 0;
			nextImage[pixel] = depthImage[pixel];
		}
	}
   
	for(iteration = 0; iteration < maxIterations; iteration++) {

		error = 0;
        for(int y = 0; y < rows; y++) {
            for(int x = 0; x < cols; x++) {

                int pixel = y * cols + x;
                if(scribbleImage[pixel] == 255) continue;
                
                float count = 0;
                float sum = 0;
                for(int neighbour = 0; neighbour < 4; neighbour++) {
                    if(positions[pixel * 4 + neighbour] != -1) {
                        sum += weights[pixel * 4 + neighbour] * image[positions[pixel * 4 + neighbour]];
                        count += weights[pixel * 4 + neighbour];
                    }
                }

				//Successive Over-Relaxation (SOR) for Gauss-Seidel only
				if(count > 0) {
                    error += fabs(sum/count - image[pixel]);
					if(solverMethod == "CPU-GaussSeidel") image[pixel] = theta * (sum / count - image[pixel]) + image[pixel];
					else nextImage[pixel] = sum/count;
                }
                
            }
        }
        
		if (solverMethod == "CPU-Jacobi-Chebyshev") {
			
			if (iteration < S) omega = 1;
			else if (iteration == S) omega = 2.0 / (2.0 - rho * rho);
			else omega = 4.0 / (4.0 - rho * rho * omega);

			for (int pixel = 0; pixel < rows * cols; pixel++) {
				if (scribbleImage[pixel] != 255)
					nextImage[pixel] = (omega * (gamma * (nextImage[pixel] - image[pixel]) + image[pixel] - previousImage[pixel])) + previousImage[pixel];
				previousImage[pixel] = image[pixel];
				image[pixel] = nextImage[pixel];
			}

		} else if (solverMethod == "CPU-Jacobi") {
		
			for (int pixel = 0; pixel < rows * cols; pixel++)
				image[pixel] = nextImage[pixel];
		
		}

        error /= (rows * cols);
        if(error < threshold) break;
    
    }

	if (isDebugEnabled) std::cout << "Iterations: " << iteration << " | Error: " << error << std::endl;
    for(int pixel = 0; pixel < rows * cols; pixel++)
        depthImage[pixel] = image[pixel];

	delete [] previousImage;
    delete [] nextImage;

}

void Solver::runConjugateGradient(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{

	Eigen::initParallel();
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(rows * cols, rows * cols);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows * cols);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(rows * cols);

	std::vector<Eigen::Triplet<double> > tripletList;
    tripletList.reserve(rows * cols * 5);
	
    for(int pixel = 0; pixel < rows * cols; pixel++)
        x(pixel) = depthImage[pixel];
	
	for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

            int pixel = y * cols + x;
            if(scribbleImage[pixel] == 255) { 
                tripletList.push_back(Eigen::Triplet<double>(pixel, pixel, 1));
                b(pixel) = depthImage[pixel];
                continue;
            }

            float sum = 0;
            if(x < cols - 1) { 
				tripletList.push_back(Eigen::Triplet<double>(pixel, y * cols + x + 1, -weights[pixel * 4 + 1]));
                sum += weights[pixel * 4 + 1];
            }
            if(x > 0) { 
                tripletList.push_back(Eigen::Triplet<double>(pixel, y * cols + x - 1, -weights[pixel * 4 + 0]));
                sum += weights[pixel * 4 + 0];
            }
            if(y < rows - 1) {
                tripletList.push_back(Eigen::Triplet<double>(pixel, (y + 1) * cols + x, -weights[pixel * 4 + 3]));
                sum += weights[pixel * 4 + 3];
            }
            if(y > 0) { 
                tripletList.push_back(Eigen::Triplet<double>(pixel, (y - 1) * cols + x, -weights[pixel * 4 + 2]));
                sum += weights[pixel * 4 + 2];
            }
            tripletList.push_back(Eigen::Triplet<double>(pixel, pixel, sum));
            
        }
    }
	
	
    A.setFromTriplets(tripletList.begin(), tripletList.end());
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> cg;
	//cg.preconditioner().setDroptol(1e-03);
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> cg;
	cg.compute(A);
	cg.setMaxIterations(maxIterations);
    cg.setTolerance(threshold);
	x = cg.solveWithGuess(b, x);
	if(isDebugEnabled) std::cout << "Iterations: " << cg.iterations() << " | Error: " << cg.error() << std::endl;
	
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = x(pixel);
    
}


void Solver::runAMG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{

	Epetra_SerialComm comm;
	const int matrixSize = rows * cols;

	Epetra_Map map(matrixSize, 0, comm);
	int *myGlobalElements = map.MyGlobalElements();

	Epetra_CrsMatrix A(Copy, map, 5);
	Epetra_Vector x(map);
	Epetra_Vector b(map);

	computePositions(rows, cols);
	computeWeights(grayImage, depthImage, maxLevel, rows, cols);

	double w[5];
	int indices[5];
	int i = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int pixel = y * cols + x;
			double sum = 0;
			int count = 0;
			if (scribbleImage[pixel] == 255) {
				w[0] = 1;
				indices[0] = pixel;
				A.InsertGlobalValues(myGlobalElements[i], 1, w, indices);
				b[pixel] = depthImage[pixel];
				i++;
				continue;
			}
			if (x < cols - 1) {
				w[count] = -weights[pixel * 4 + 1];
				indices[count] = y * cols + x + 1;
				sum += -w[count];
				count++;
			}
			if (x > 0) {
				w[count] = -weights[pixel * 4 + 0];
				indices[count] = y * cols + x - 1;
				sum += -w[count];
				count++;
			}
			if (y < rows - 1) {
				w[count] = -weights[pixel * 4 + 3];
				indices[count] = (y + 1) * cols + x;
				sum += -w[count];
				count++;
			}
			if (y > 0) {
				w[count] = -weights[pixel * 4 + 2];
				indices[count] = (y - 1) * cols + x;
				sum += -w[count];
				count++;
			}
			w[count] = sum;
			indices[count] = pixel;
			count++;
			A.InsertGlobalValues(myGlobalElements[i], count, w, indices);
			i++;
		}
	}

	for (int pixel = 0; pixel < rows * cols; pixel++)
		x[pixel] = depthImage[pixel];

	A.FillComplete();

	Epetra_LinearProblem problem(&A, &x, &b);
	AztecOO solver(problem);

	ML_Epetra::MultiLevelPreconditioner MLPrec(A, true);
	
	solver.SetPrecOperator(&MLPrec);
	solver.SetAztecOption(AZ_solver, AZ_bicgstab);
	solver.SetAztecOption(AZ_precond, AZ_Jacobi);
	solver.SetAztecOption(AZ_diagnostics, AZ_none);
	solver.SetAztecOption(AZ_output, AZ_none);
	solver.Iterate(maxIterations, threshold);
	if(isDebugEnabled) std::cout << "Iterations: " << solver.NumIters() << " | Error: " << solver.TrueResidual() << std::endl;

	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = x[pixel];

}

void Solver::runLAHBPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{
	
	Eigen::MatrixXd d = Eigen::MatrixXd(cols, rows);
	Eigen::MatrixXd gx = Eigen::MatrixXd::Ones(cols, rows);
	Eigen::MatrixXd gy = Eigen::MatrixXd::Ones(cols, rows);
	Eigen::MatrixXd sx = Eigen::MatrixXd::Ones(cols, rows);
	Eigen::MatrixXd sy = Eigen::MatrixXd::Ones(cols, rows);
	Eigen::MatrixXd w = Eigen::MatrixXd::Ones(cols, rows);
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int pixel = y * cols + x;
			d(x, y) = depthImage[pixel];
			w(x, y) = scribbleImage[pixel] / 255;
		}
	}

	LAHBPCG solver;
	Eigen::MatrixXd xV = solver.apply(d, gx, gy, w, sx, sy, maxIterations, threshold);
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int pixel = y * cols + x;
			depthImage[pixel] = xV(x, y);
		}
	}
	
}

void Solver::computeWeights(unsigned char *grayImage, unsigned char *depthImage, int level, int rows, int cols)
{

    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

			int pixel = y * cols + x;

			if (method == "Guo" || (method == "Macedo" && level == maxLevel)) {
			
				for (int neighbour = 0; neighbour < 4; neighbour++)
					if (positions[pixel * 4 + neighbour] != -1)
						weights[pixel * 4 + neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[pixel * 4 + neighbour]]));
			
			} else if (method == "Macedo" && level != 0) {

				for (int neighbour = 0; neighbour < 4; neighbour++) {
					if (positions[pixel * 4 + neighbour] != -1) {
						if (edges[pixel] != 0 && edges[positions[pixel * 4 + neighbour]] != 0) weights[pixel * 4 + neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[pixel * 4 + neighbour]]));
						else weights[pixel * 4 + neighbour] = expf(-beta * fabs(depthImage[pixel] - depthImage[positions[pixel * 4 + neighbour]]));
					}
				}

			} else {

				for (int neighbour = 0; neighbour < 4; neighbour++)
					if (positions[pixel * 4 + neighbour] != -1)
						weights[pixel * 4 + neighbour] = expf(-beta * fabs(depthImage[pixel] - depthImage[positions[pixel * 4 + neighbour]]));

			}

        }
    }

}

void Solver::computePositions(int rows, int cols)
{

    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

            int pixel = y * cols + x;
            for(int neighbour = 0; neighbour < 4; neighbour++)
                positions[pixel * 4 + neighbour] = -1;
                
            if(x - 1 >= 0) positions[pixel * 4 + 0] = y * cols + x - 1;
            if(x + 1 < cols) positions[pixel * 4 + 1] = y * cols + x + 1;
            if(y - 1 >= 0) positions[pixel * 4 + 2] = (y - 1) * cols + x;
            if(y + 1 < rows) positions[pixel * 4 + 3] = (y + 1) * cols + x;

        }
    }

}

void Solver::computeEdges(unsigned char *depthImage, int rows, int cols)
{

	for (int pixel = 0; pixel < rows * cols; pixel++)
		edges[pixel] = 0;

	for (int x = 1; x < cols - 1; x++) {
		for (int y = 1; y < rows - 1; y++) {

			int center = depthImage[y * cols + x];
			int top = depthImage[(y - 1) * cols + x];
			int bottom = depthImage[(y + 1) * cols + x];
			int left = depthImage[y * cols + x - 1];
			int right = depthImage[y * cols + x + 1];

			if (abs(center - top) > 4 || abs(center - bottom) > 4 || abs(center - left) > 4 || abs(center - right) > 4) edges[y * cols + x] = 255;
			else edges[y * cols + x] = 0;

		}
	}

}