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


    for(int pixel = 0; pixel < rows * cols; pixel++) {
        image[pixel] = depthImage[pixel];
        tempImage[pixel] = image[pixel];
    }
    
    for(int iteration = 0; iteration < maxIterations; iteration++) {

        float error = 0;
        int positions[4];
        float neighbours[4];
        float weights[4];       
        
        for(int y = 0; y < rows; y++) {
            for(int x = 0; x < cols; x++) {

                int pixel = y * cols + x;
                if(scribbleImage[pixel] == 255) continue;
                
                for(int neighbour = 0; neighbour < 4; neighbour++)
                    positions[neighbour] = -1;
                
                if(x - 1 >= 0) positions[0] = y * cols + x - 1;
                if(x + 1 < cols) positions[1] = y * cols + x + 1;
                if(y - 1 >= 0) positions[2] = (y - 1) * cols + x;
                if(y + 1 < rows) positions[3] = (y + 1) * cols + x;

                //set all weights to 1 to suppress gradient constraint
                float count = 0;
                float sum = 0;
                for(int neighbour = 0; neighbour < 4; neighbour++) {
                    if(positions[neighbour] != -1) {
                        weights[neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[neighbour]]));
                        sum += weights[neighbour] * image[positions[neighbour]];
                        count += weights[neighbour];
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

}

void Solver::runGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{
    
    for(int pixel = 0; pixel < rows * cols; pixel++)
        image[pixel] = depthImage[pixel];
    
    float error = 0;
    
    for(int iteration = 0; iteration < maxIterations; iteration++) {

        error = 0;
        int positions[4];
        float neighbours[4];
        float weights[4];       
        
        for(int y = 0; y < rows; y++) {
            for(int x = 0; x < cols; x++) {

                int pixel = y * cols + x;
                if(scribbleImage[pixel] == 255) continue;
                
                for(int neighbour = 0; neighbour < 4; neighbour++)
                    positions[neighbour] = -1;
                
                if(x - 1 >= 0) positions[0] = y * cols + x - 1;
                if(x + 1 < cols) positions[1] = y * cols + x + 1;
                if(y - 1 >= 0) positions[2] = (y - 1) * cols + x;
                if(y + 1 < rows) positions[3] = (y + 1) * cols + x;

                //set all weights to 1 to suppress gradient constraint
                float count = 0;
                float sum = 0;
                for(int neighbour = 0; neighbour < 4; neighbour++) {
                    if(positions[neighbour] != -1) {
                        weights[neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[neighbour]]));
                        sum += weights[neighbour] * image[positions[neighbour]];
                        count += weights[neighbour];
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
            std::cout << "Iteration: " << iteration << ", Error: " << error << std::endl;
            cv::imshow("Depth Image", debugImage);
            cv::waitKey(33);
        }
        
    }

    std::cout << error << std::endl;
    for(int pixel = 0; pixel < rows * cols; pixel++)
        depthImage[pixel] = image[pixel];

}

void Solver::runConjugateGradient(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols)
{

    Eigen::SparseMatrix<double> A(rows * cols, rows * cols);
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
    
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A.transpose()*A);
    cg.setMaxIterations(maxIterations);
    cg.setTolerance(threshold);
    x = cg.solveWithGuess(A.transpose() * b, x);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;

    for(int pixel = 0; pixel < rows * cols; pixel++) {
        if(x(pixel) < 0) depthImage[pixel] = 0;
        else depthImage[pixel] = x(pixel);
    }
    
}
