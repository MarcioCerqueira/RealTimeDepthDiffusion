#include "Solver.h"

Solver::Solver(int rows, int cols)
{
    
    image = (float*)malloc(sizeof(float) * rows * cols);
    debugImage = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
    this->rows = rows;
    this->cols = cols;
    this->isDebugEnabled = false;

}

Solver::~Solver()
{
    delete [] image;
}

void Solver::runJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{

    float *tempImage = (float*)malloc(sizeof(float) * rows * cols);
    float *weights = (float*)malloc(rows * cols * 4 * sizeof(float));
    int *positions = (int*)malloc(rows * cols * 4 * sizeof(int));
    computeWeights(weights, grayImage, rows, cols);
    computePositions(positions, rows, cols);

    for(int pixel = 0; pixel < rows * cols; pixel++) {
        image[pixel] = depthImage[pixel];
        tempImage[pixel] = image[pixel];
    }
    
    for(int iteration = 0; iteration < maxIterations; iteration++) {

        float error = 0;
        
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

                if(count > 0) {
                    error += fabs(sum/count - image[pixel]);
                    tempImage[pixel] = sum/count;
                }
                
            }
        }
        
        for(int pixel = 0; pixel < rows * cols; pixel++)
            image[pixel] = tempImage[pixel];

        error /= (rows * cols);
        if(error < threshold) break;
        
        if(iteration % 10 == 0 && isDebugEnabled) {
            for(int pixel = 0; pixel < rows * cols; pixel++) {
                if(image[pixel] == -1) debugImage.ptr<unsigned char>()[pixel] = 255;
                else debugImage.ptr<unsigned char>()[pixel] = image[pixel];      
            }  
            std::cout << "Iteration: " << iteration << ", Error: " << error << std::endl;
            cv::imshow("Depth Image", debugImage);
            cv::waitKey(33);
        }
        
    }

    for(int pixel = 0; pixel < rows * cols; pixel++)
        depthImage[pixel] = image[pixel];


    delete [] tempImage;
    delete [] weights;
    delete [] positions;

}

void Solver::runGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{
    
    for(int pixel = 0; pixel < rows * cols; pixel++)
        image[pixel] = depthImage[pixel];
    
    float error = 0;
    float *weights = (float*)malloc(rows * cols * 4 * sizeof(float));
    int *positions = (int*)malloc(rows * cols * 4 * sizeof(int));
    computeWeights(weights, grayImage, rows, cols);
    computePositions(positions, rows, cols);

    for(int iteration = 0; iteration < maxIterations; iteration++) {

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

                if(count > 0) {
                    error += fabs(sum/count - image[pixel]);
                    image[pixel] = sum/count;
                }
                
            }
        }
        
        error /= (rows * cols);
        if(error < threshold) break;
        
        if(iteration % 10 == 0 && isDebugEnabled) {
            for(int pixel = 0; pixel < rows * cols; pixel++) {
                if(image[pixel] == -1) debugImage.ptr<unsigned char>()[pixel] = 255;
                else debugImage.ptr<unsigned char>()[pixel] = image[pixel];      
            }  
            //std::cout << "Iteration: " << iteration << ", Error: " << error << std::endl;
            cv::imshow("Depth Image", debugImage);
            cv::waitKey(33);
        }
        
    }

    std::cout << error << std::endl;
    for(int pixel = 0; pixel < rows * cols; pixel++)
        depthImage[pixel] = image[pixel];

    delete [] weights;
    delete [] positions;

}

void Solver::runConjugateGradient(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{

	Eigen::initParallel();
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(rows * cols, rows * cols);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows * cols);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(rows * cols);

    std::vector<Eigen::Triplet<double> > tripletList;
    tripletList.reserve(rows * cols * 5);
    float weight;

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
                weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));   
                tripletList.push_back(Eigen::Triplet<double>(pixel, y * cols + x + 1, -weight)); 
                sum += weight; 
            }
            if(x > 0) { 
                weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));   
                tripletList.push_back(Eigen::Triplet<double>(pixel, y * cols + x - 1, -weight)); 
                sum += weight; 
            }
            if(y < rows - 1) {
                weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));   
                tripletList.push_back(Eigen::Triplet<double>(pixel, (y + 1) * cols + x, -weight));
                sum += weight;
            }
            if(y > 0) { 
                weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));   
                tripletList.push_back(Eigen::Triplet<double>(pixel, (y - 1) * cols + x, -weight)); 
                sum += weight; 
            }
            tripletList.push_back(Eigen::Triplet<double>(pixel, pixel, sum));
            
        }
    }
	
	
    A.setFromTriplets(tripletList.begin(), tripletList.end());
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> cg;
	cg.compute(A);
	cg.setMaxIterations(maxIterations);
    cg.setTolerance(threshold);
	x = cg.solveWithGuess(b, x);
	std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		depthImage[pixel] = x(pixel);
	}
    
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

	double weights[5];
	int indices[5];
	int i = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int pixel = y * cols + x;
			double sum = 0;
			int count = 0;
			if (scribbleImage[pixel] == 255) {
				weights[0] = 1;
				indices[0] = pixel;
				A.InsertGlobalValues(myGlobalElements[i], 1, weights, indices);
				b[pixel] = depthImage[pixel];
				i++;
				continue;
			}
			if (x < cols - 1) {
				weights[count] = -expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				indices[count] = y * cols + x + 1;
				sum += -weights[count];
				count++;
			}
			if (x > 0) {
				weights[count] = -expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				indices[count] = y * cols + x - 1;
				sum += -weights[count];
				count++;
			}
			if (y < rows - 1) {
				weights[count] = -expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				indices[count] = (y + 1) * cols + x;
				sum += -weights[count];
				count++;
			}
			if (y > 0) {
				weights[count] = -expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				indices[count] = (y - 1) * cols + x;
				sum += -weights[count];
				count++;
			}
			weights[count] = sum;
			indices[count] = pixel;
			count++;
			A.InsertGlobalValues(myGlobalElements[i], count, weights, indices);
			i++;
		}
	}

	for (int pixel = 0; pixel < rows * cols; pixel++)
		x[pixel] = depthImage[pixel];

	A.FillComplete();
	/*
	//Belos version
	Teuchos::RCP<Epetra_Map> belosMap = Teuchos::rcp(&map, false);
	Teuchos::RCP<Epetra_CrsMatrix> belosA = Teuchos::rcp(&A, false);
	Teuchos::RCP<Epetra_Vector> belosVecB = Teuchos::rcp(&b, false);
	Teuchos::RCP<Epetra_Vector> belosVecX = Teuchos::rcp(&x, false);
	Teuchos::RCP<Epetra_MultiVector> belosB = Teuchos::rcp_implicit_cast<Epetra_MultiVector>(belosVecB);
	Teuchos::RCP<Epetra_MultiVector> belosX = Teuchos::rcp_implicit_cast<Epetra_MultiVector>(belosVecX);

	belosA->OptimizeStorage();
	Teuchos::ParameterList belosList;
	belosList.set("Block Size", 1);
	belosList.set("Maximum Iterations", maxIterations * 3);
	belosList.set("Convergence Tolerance", 1e-07);
	//belosList.set("Verbosity", Belos::Errors + Belos::Warnings +
	//	Belos::TimingDetails + Belos::FinalSummary +
	//	Belos::StatusTestDetails);
	Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator> problem(belosA, belosX, belosB);
	bool set = problem.setProblem();
	Teuchos::RCP< Belos::SolverManager<double, Epetra_MultiVector, Epetra_Operator> > solver =
	Teuchos::rcp(new Belos::BlockCGSolMgr<double, Epetra_MultiVector, Epetra_Operator>(Teuchos::rcp(&problem, false), Teuchos::rcp(&belosList, false)));
	solver->solve();

	for (int pixel = 0; pixel < rows * cols; pixel++)
	depthImage[pixel] = belosX->Pointers()[0][pixel];
	*/

	/*
	// AztecOO version
	Epetra_LinearProblem problem(&A, &x, &b);
	AztecOO solver(problem);

	solver.SetAztecOption(AZ_solver, AZ_bicgstab);
	solver.SetAztecOption(AZ_precond, AZ_Jacobi);
	solver.SetAztecOption(AZ_diagnostics, AZ_none);
	solver.SetAztecOption(AZ_output, AZ_none);
	solver.Iterate(maxIterations, threshold);
	std::cout << "#iterations:     " << solver.NumIters() << std::endl;
	std::cout << "estimated error: " << solver.TrueResidual() << std::endl;

	for (int pixel = 0; pixel < rows * cols; pixel++)
	depthImage[pixel] = x[pixel];
	*/

	// AztecOO version
	Epetra_LinearProblem problem(&A, &x, &b);
	AztecOO solver(problem);

	ML_Epetra::MultiLevelPreconditioner MLPrec(A, true);

	solver.SetPrecOperator(&MLPrec);
	solver.SetAztecOption(AZ_solver, AZ_bicgstab);
	solver.SetAztecOption(AZ_precond, AZ_Jacobi);
	solver.SetAztecOption(AZ_diagnostics, AZ_none);
	solver.SetAztecOption(AZ_output, AZ_none);
	solver.Iterate(maxIterations, threshold);
	std::cout << "#iterations:     " << solver.NumIters() << std::endl;
	std::cout << "estimated error: " << solver.TrueResidual() << std::endl;

	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = x[pixel];

}

void Solver::computeWeights(float *weights, unsigned char *grayImage, int rows, int cols)
{

    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

            int positions[4];
            int pixel = y * cols + x;
            for(int neighbour = 0; neighbour < 4; neighbour++)
                positions[neighbour] = -1;
                
            if(x - 1 >= 0) positions[0] = y * cols + x - 1;
            if(x + 1 < cols) positions[1] = y * cols + x + 1;
            if(y - 1 >= 0) positions[2] = (y - 1) * cols + x;
            if(y + 1 < rows) positions[3] = (y + 1) * cols + x;

            for(int neighbour = 0; neighbour < 4; neighbour++) {
                if(positions[neighbour] != -1) {
                    weights[pixel * 4 + neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[neighbour]]));
                }
            }

        }
    }

}

void Solver::computePositions(int *positions, int rows, int cols)
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