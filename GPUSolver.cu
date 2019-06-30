#include <cuda_runtime.h>
#include <iostream>
#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/relaxation/gauss_seidel.h>
#include <cusp/relaxation/jacobi.h>
#include <paralution.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/amg.hpp>
#include <time.h>
#include "GPUSolver.h"

bool isParalutionInitialized = false;

void GPUCheckError2(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

int divUp(int a, int b) { 
    return (a + b - 1)/b;
}

__global__ void jacobi(float *output, float *input, float *weights, int *positions, unsigned char *scribbleImage, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int pixel = y * cols + x;
	if(scribbleImage[pixel] == 255) return;
	            
    float count = 0;
    float sum = 0;
	float weight = 0;
	int position = 0;
    for(int neighbour = 0; neighbour < 4; neighbour++) {
		position = positions[pixel * 4 + neighbour];
		if(position != -1) {
			weight = weights[pixel * 4 + neighbour];
			sum += weight * input[position];
			count += weight;
		}
	}
    
	if(count > 0) output[pixel] = sum/count;

}

__global__ void gaussSeidel(float *image, float *weights, int *colors, int *positions, unsigned char *scribbleImage, int color, int rows, int cols)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	int pixel = y * cols + x;
	if(scribbleImage[pixel] == 255) return;
	if(colors[pixel] != color) return;
	            
    float count = 0;
    float sum = 0;
	float weight = 0;
	int position = 0;
    for(int neighbour = 0; neighbour < 4; neighbour++) {
		position = positions[pixel * 4 + neighbour];
		if(position != -1) {
			weight = weights[pixel * 4 + neighbour];
			sum += weight * image[position];
			count += weight;
		}
	}
    
	if(count > 0) image[pixel] = sum/count;

}

void GPUJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	float beta, int maxIterations, float tolerance)
{

	float *tempImage = (float*)malloc(sizeof(float) * rows * cols);
	float *image = (float*)malloc(sizeof(float) * rows * cols);
	float *weights = (float*)malloc(rows * cols * 4 * sizeof(float));
    int *positions = (int*)malloc(rows * cols * 4 * sizeof(int));
    int iteration;

	for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

            int pixel = y * cols + x;
            for(int neighbour = 0; neighbour < 4; neighbour++)
                positions[pixel * 4 + neighbour] = -1;
                
            if(x - 1 >= 0) positions[pixel * 4 + 0] = y * cols + x - 1;
            if(x + 1 < cols) positions[pixel * 4 + 1] = y * cols + x + 1;
            if(y - 1 >= 0) positions[pixel * 4 + 2] = (y - 1) * cols + x;
            if(y + 1 < rows) positions[pixel * 4 + 3] = (y + 1) * cols + x;

            for(int neighbour = 0; neighbour < 4; neighbour++)
                if(positions[pixel * 4 + neighbour] != -1)
                    weights[pixel * 4 + neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[pixel * 4 + neighbour]]));

        }
    }

	for(int pixel = 0; pixel < rows * cols; pixel++) {
        image[pixel] = depthImage[pixel];
        tempImage[pixel] = image[pixel];
    }

	float *deviceTempImage = (float*)malloc(sizeof(float) * rows * cols);
	float *deviceImage = (float*)malloc(sizeof(float) * rows * cols);
	float *deviceWeights = (float*)malloc(rows * cols * 4 * sizeof(float));
    int *devicePositions = (int*)malloc(rows * cols * 4 * sizeof(int));
    unsigned char *deviceScribbleImage = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	
	cudaMalloc((void**)&deviceTempImage, sizeof(float) * rows * cols);
    cudaMalloc((void**)&deviceImage, sizeof(float) * rows * cols);
    cudaMalloc((void**)&deviceWeights, sizeof(float) * rows * cols * 4);
    cudaMalloc((void**)&devicePositions, sizeof(int) * rows * cols * 4);
    cudaMalloc((void**)&deviceScribbleImage, sizeof(unsigned char) * rows * cols);
    
	cudaMemcpy(deviceTempImage, tempImage, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceImage, image, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWeights, weights, sizeof(float) * rows * cols * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(devicePositions, positions, sizeof(int) * rows * cols * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceScribbleImage, scribbleImage, sizeof(unsigned char) * rows * cols, cudaMemcpyHostToDevice);
	
	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

	for(iteration = 0; iteration < maxIterations; iteration++) {
		jacobi<<<grid, threads>>>(deviceTempImage, deviceImage, deviceWeights, devicePositions, deviceScribbleImage, rows, cols);
		cudaMemcpy(deviceImage, deviceTempImage, sizeof(float) * rows * cols, cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(image, deviceImage, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    
	for(int pixel = 0; pixel < rows * cols; pixel++)
        depthImage[pixel] = image[pixel];

	cudaFree(deviceTempImage);
	cudaFree(deviceImage);
	cudaFree(deviceWeights);
	cudaFree(devicePositions);
	
	delete [] image;
	delete [] tempImage;
	delete [] weights;
	delete [] positions;

}


void GPUGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	float beta, int maxIterations, float tolerance)
{

	float *image = (float*)malloc(sizeof(float) * rows * cols);
	float *weights = (float*)malloc(rows * cols * 4 * sizeof(float));
    int *colors = (int*)malloc(sizeof(int) * rows * cols);
	int *positions = (int*)malloc(rows * cols * 4 * sizeof(int));
    int iteration;

	for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

            int pixel = y * cols + x;
            for(int neighbour = 0; neighbour < 4; neighbour++)
                positions[pixel * 4 + neighbour] = -1;
                
            if(x - 1 >= 0) positions[pixel * 4 + 0] = y * cols + x - 1;
            if(x + 1 < cols) positions[pixel * 4 + 1] = y * cols + x + 1;
            if(y - 1 >= 0) positions[pixel * 4 + 2] = (y - 1) * cols + x;
            if(y + 1 < rows) positions[pixel * 4 + 3] = (y + 1) * cols + x;

            for(int neighbour = 0; neighbour < 4; neighbour++)
                if(positions[pixel * 4 + neighbour] != -1)
                    weights[pixel * 4 + neighbour] = expf(-beta * fabs(grayImage[pixel] - grayImage[positions[pixel * 4 + neighbour]]));

			if(x % 2 == 0 && y % 2 == 0) colors[pixel] = 0;
			else if(x % 2 == 0 && y % 2 == 1) colors[pixel] = 1;
			else if(x % 2 == 1 && y % 2 == 0) colors[pixel] = 2;
			else colors[pixel] = 3;

        }
    }

	for(int pixel = 0; pixel < rows * cols; pixel++)
        image[pixel] = depthImage[pixel];

	float *deviceImage = (float*)malloc(sizeof(float) * rows * cols);
	float *deviceWeights = (float*)malloc(rows * cols * 4 * sizeof(float));
    int *deviceColors = (int*)malloc(rows * cols * sizeof(int));
	int *devicePositions = (int*)malloc(rows * cols * 4 * sizeof(int));
    unsigned char *deviceScribbleImage = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	
	cudaMalloc((void**)&deviceImage, sizeof(float) * rows * cols);
    cudaMalloc((void**)&deviceWeights, sizeof(float) * rows * cols * 4);
    cudaMalloc((void**)&deviceColors, sizeof(int) * rows * cols);
    cudaMalloc((void**)&devicePositions, sizeof(int) * rows * cols * 4);
    cudaMalloc((void**)&deviceScribbleImage, sizeof(unsigned char) * rows * cols);
    
	cudaMemcpy(deviceImage, image, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWeights, weights, sizeof(float) * rows * cols * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceColors, colors, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
	cudaMemcpy(devicePositions, positions, sizeof(int) * rows * cols * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceScribbleImage, scribbleImage, sizeof(unsigned char) * rows * cols, cudaMemcpyHostToDevice);
	
	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	int maxColors = 4;

	for(iteration = 0; iteration < maxIterations; iteration++)
		for(int color = 0; color < maxColors; color++)
			gaussSeidel<<<grid, threads>>>(deviceImage, deviceWeights, deviceColors, devicePositions, deviceScribbleImage, color, rows, cols);

	cudaMemcpy(image, deviceImage, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    
	for(int pixel = 0; pixel < rows * cols; pixel++)
        depthImage[pixel] = image[pixel];

	cudaFree(deviceImage);
	cudaFree(deviceWeights);
	cudaFree(deviceColors);
	cudaFree(devicePositions);
	
	delete [] image;
	delete [] weights;
	delete [] colors;
	delete [] positions;

}
/*
double cpuTime2(void)
{

	double value;
	value = (double) clock () / (double) CLOCKS_PER_SEC;
	return value;

}
*/

void CUSPJacobi(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	float beta, int maxIterations, float tolerance)
{

    cusp::csr_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5);
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	float weight;
	int counter = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			A.row_offsets[pixel] = counter;

			if (scribbleImage[pixel] == 255) {
				A.column_indices[counter] = pixel;
				A.values[counter] = 1;
				b[pixel] = depthImage[pixel];
				counter++;
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				A.column_indices[counter] = (y - 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				A.column_indices[counter] = y * cols + x - 1;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				A.column_indices[counter] = y * cols + x + 1;
				A.values[counter] = -weight;
				sum += weight;
				counter++;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				A.column_indices[counter] = (y + 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			
			A.column_indices[counter] = pixel;
			A.values[counter] = sum;
			counter++;

		}
	}
	A.row_offsets[rows * cols] = counter;

	cusp::array1d<float, cusp::device_memory> d_x(x);
	cusp::array1d<float, cusp::device_memory> d_b(b);
	
	cusp::csr_matrix<int, float, cusp::device_memory> d_A(A);
	cusp::relaxation::jacobi<float, cusp::device_memory> M(d_A);
	cusp::array1d<float, cusp::device_memory> d_r(A.num_rows);
	cusp::multiply(d_A, d_x, d_r);
	cusp::blas::axpy(d_b, d_r, float(-1));
	
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, 0, false);
	
	while (!monitor.finished(d_r))
	{
		M(d_A, d_b, d_x);
		cusp::multiply(d_A, d_x, d_r);
		cusp::blas::axpy(d_b, d_r, float(-1));
		++monitor;
	}
	
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];
	
}

void CUSPGaussSeidel(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, 
	float beta, int maxIterations, float tolerance)
{

    cusp::csr_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5);
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	float weight;
	int counter = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			A.row_offsets[pixel] = counter;

			if (scribbleImage[pixel] == 255) {
				A.column_indices[counter] = pixel;
				A.values[counter] = 1;
				b[pixel] = depthImage[pixel];
				counter++;
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				A.column_indices[counter] = (y - 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				A.column_indices[counter] = y * cols + x - 1;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				A.column_indices[counter] = y * cols + x + 1;
				A.values[counter] = -weight;
				sum += weight;
				counter++;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				A.column_indices[counter] = (y + 1) * cols + x;
				A.values[counter] = -weight;
				counter++;
				sum += weight;
			}
			
			A.column_indices[counter] = pixel;
			A.values[counter] = sum;
			counter++;

		}
	}
	A.row_offsets[rows * cols] = counter;

	cusp::array1d<float, cusp::device_memory> d_x(x);
	cusp::array1d<float, cusp::device_memory> d_b(b);
	
	cusp::csr_matrix<int, float, cusp::device_memory> d_A(A);
	cusp::relaxation::gauss_seidel<float, cusp::device_memory> M(d_A);
	cusp::array1d<float, cusp::device_memory> d_r(A.num_rows);
	cusp::multiply(d_A, d_x, d_r);
	cusp::blas::axpy(d_b, d_r, float(-1));
	
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, 0, false);
	
	while (!monitor.finished(d_r))
	   {
		   M(d_A, d_b, d_x);
		   cusp::multiply(d_A, d_x, d_r);
		   cusp::blas::axpy(d_b, d_r, float(-1));
		   ++monitor;
	   }
	
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];
	
}

void CUSPPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, float beta, 
	int maxIterations, float tolerance)
{
	
	cusp::ell_matrix<int, float, cusp::host_memory> A(rows * cols, rows * cols, rows * cols * 5, 5);
	const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
	cusp::array1d<float, cusp::host_memory> x(A.num_rows);
	cusp::array1d<float, cusp::host_memory> b(A.num_rows);
	
	for (int pixel = 0; pixel < rows * cols; pixel++) {
		x[pixel] = depthImage[pixel];
		b[pixel] = 0;
	}

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int pixel = y * cols + x;
			for(int w = 0; w < 5; w++) {
				A.column_indices(pixel, w) = X;
				A.values(pixel, w) = 0;
			}
		}
	}

	float weight;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int counter = 0;
			int pixel = y * cols + x;
			if (scribbleImage[pixel] == 255) {
				A.column_indices(pixel, 0) = pixel;
				A.values(pixel, 0) = 1;
				b[pixel] = depthImage[pixel];
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				A.column_indices(pixel, counter) = (y - 1) * cols + x;
				A.values(pixel, counter) = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				A.column_indices(pixel, counter) = y * cols + x - 1;
				A.values(pixel, counter) = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				A.column_indices(pixel, counter) = y * cols + x + 1;
				A.values(pixel, counter) = -weight;
				sum += weight;
				counter++;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				A.column_indices(pixel, counter) = (y + 1) * cols + x;
				A.values(pixel, counter) = -weight;
				counter++;
				sum += weight;
			}
			
			A.column_indices(pixel, counter) = pixel;
			A.values(pixel, counter) = sum;

		}
	}

	cusp::ell_matrix<int, float, cusp::device_memory> d_A(A);
	cusp::array1d<float, cusp::device_memory> d_x(x);
	cusp::array1d<float, cusp::device_memory> d_b(b);
	cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_rows);
	cusp::monitor<float> monitor(d_b, maxIterations, tolerance, 0, false);
	cusp::krylov::bicgstab(d_A, d_x, d_b, monitor, M);
	
	cusp::array1d<float, cusp::host_memory> r(d_x);
	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = r[pixel];

}

void ParalutionPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, float beta, 
	int maxIterations, float tolerance)
{

	if(!isParalutionInitialized) {
		paralution::init_paralution();
		isParalutionInitialized = true;
	}
	
	float *h_x = (float*)malloc(rows * cols * sizeof(float));
	float *h_b = (float*)malloc(rows * cols * sizeof(float));
	int *row_offsets = (int*)malloc((rows * cols + 1) * sizeof(int));
	int *col = (int*)malloc(rows * cols * 5 * sizeof(int));
	float *val = (float*)malloc(rows * cols * 5 * sizeof(float));

	for (int pixel = 0; pixel < rows * cols; pixel++) {
		h_x[pixel] = depthImage[pixel];
		h_b[pixel] = 0;
	}

	float weight;
	int counter = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			row_offsets[pixel] = counter;

			if (scribbleImage[pixel] == 255) {
				col[counter] = pixel;
				val[counter] = 1;
				h_b[pixel] = depthImage[pixel];
				counter++;
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				col[counter] = (y - 1) * cols + x;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				col[counter] = y * cols + x - 1;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				col[counter] = y * cols + x + 1;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				col[counter] = (y + 1) * cols + x;
				val[counter] = -weight;
				counter++;
				sum += weight;
			}
			
			col[counter] = pixel;
			val[counter] = sum;
			counter++;

		}
	}
	row_offsets[rows * cols] = counter;

	paralution::LocalMatrix<float> A;
	paralution::LocalVector<float> x;
	paralution::LocalVector<float> b;

	A.AllocateCSR("A", rows * cols * 5, rows * cols, rows * cols);
	A.CopyFromCSR(row_offsets, col, val);
	x.Allocate("x", A.get_nrow());
	x.SetDataPtr(&h_x, "vector", A.get_nrow());
	b.Allocate("b", A.get_nrow());
	b.SetDataPtr(&h_b, "vector", A.get_nrow());
	
	paralution::BiCGStab<paralution::LocalMatrix<float>, paralution::LocalVector<float>, float > ls;
	paralution::Jacobi<paralution::LocalMatrix<float>, paralution::LocalVector<float>, float > p;
	
	A.MoveToAccelerator();
	x.MoveToAccelerator();
	b.MoveToAccelerator();
	ls.MoveToAccelerator();
	
	ls.Init(tolerance, tolerance, 1e+10, maxIterations);
	ls.SetOperator(A);
	ls.SetPreconditioner(p);
	ls.Verbose(0);
	ls.Build();
	ls.Solve(b, &x);

	x.MoveToHost();
	x.LeaveDataPtr(&h_x);

	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = h_x[pixel];

	delete [] h_x;
	delete [] h_b;
	delete [] row_offsets;
	delete [] col;
	delete [] val;

	A.Clear();
	x.Clear();
	b.Clear();
	ls.Clear();
	//paralution::stop_paralution();	
	
}

void ViennaCLPCG(unsigned char *depthImage, unsigned char *scribbleImage, unsigned char *grayImage, int rows, int cols, float beta, 
	int maxIterations, float tolerance)
{

	std::vector<float> h_x(rows * cols);
	std::vector<float> h_b(rows * cols);
	std::vector< std::map< unsigned int, float> > h_A(rows * cols);

	for (int pixel = 0; pixel < rows * cols; pixel++) {
		h_x[pixel] = depthImage[pixel];
		h_b[pixel] = 0;
	}

	float weight;
	int counter = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
		
			int pixel = y * cols + x;
			
			if (scribbleImage[pixel] == 255) {
				h_A[pixel][pixel] = 1;
				h_b[pixel] = depthImage[pixel];
				continue;
			}
			
			float sum = 0;
			if (y > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y - 1) * cols + x]));
				h_A[pixel][(y - 1) * cols + x] = -weight;
				sum += weight;
			}
			if (x > 0) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x - 1]));
				h_A[pixel][y * cols + x - 1] = -weight;
				sum += weight;
			}
			if (x < cols - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[y * cols + x + 1]));
				h_A[pixel][y * cols + x + 1] = -weight;
				sum += weight;
			}
			if (y < rows - 1) {
				weight = expf(-beta * fabs(grayImage[pixel] - grayImage[(y + 1) * cols + x]));
				h_A[pixel][(y + 1) * cols + x] = -weight;
				sum += weight;
			}
			
			h_A[pixel][pixel] = sum;

		}
	}

	viennacl::context ctx;
	viennacl::vector<float> x(rows * cols, ctx);
	viennacl::vector<float> b(rows * cols, ctx);
	viennacl::compressed_matrix<float> A(rows * cols, rows * cols, ctx);
	viennacl::copy(h_x, x);
	viennacl::copy(h_b, b);
	viennacl::copy(h_A, A);
	
	viennacl::linalg::bicgstab_tag config(tolerance, maxIterations);
	config.abs_tolerance(tolerance);
	viennacl::linalg::bicgstab_solver<viennacl::vector<float> > solver(config);
	solver.set_initial_guess(x);

	
	//viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<float> > vcl_jacobi(A, viennacl::linalg::jacobi_tag());
	//viennacl::linalg::ilu0_tag ilu0_config;
	//viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<float> > vcl_ilut(A, ilu0_config);
	//viennacl::linalg::block_ilu_precond<viennacl::compressed_matrix<float>, viennacl::linalg::ilu0_tag> vcl_block_ilu0(A, viennacl::linalg::ilu0_tag());
	//viennacl::linalg::ilut_tag ilut_config(rows * cols, 1e-03, true);
	//viennacl::linalg::ilut_precond< viennacl::compressed_matrix<float> > vcl_ilut(A, ilut_config);

	x = solver(A, b);
	viennacl::copy(x, h_x);

	for (int pixel = 0; pixel < rows * cols; pixel++)
		depthImage[pixel] = h_x[pixel];
	
}