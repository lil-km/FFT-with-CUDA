#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#include <cufft.h>
#include <cufftxt.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

typedef float2 Cplx;

#define PI 3.14159265358979323846
#define N  2048//2048

__constant__ Cplx twiddle[N / 2];

__global__ void naive_fft(Cplx* a, int n, int size);
void err_check(Cplx* b, int size);
void twiddleFactor(Cplx* w, int size);
void myCos(Cplx* a, int size);
void echo(Cplx* a, int size);
void bitReverse2natural(Cplx* a, unsigned int size, unsigned int n);
void swap(float* a, float* b);
unsigned int bitReverse(unsigned int num, unsigned int bit_num);
void cumyCos(cufftComplex* a, int size);


int main(void) {
	// host data and twiddle factor pointer
	Cplx * h_a, * h_w;
	// device data and twiddle factor pointer
	Cplx * d_a;

	// size in byte for data, multiplying by factor of two for the real and imaginary part
	unsigned int size = N * sizeof(Cplx);
	//number of iteration to perform the FFT
	unsigned int n = (unsigned int)(log(N) / log(2));

	//allocating space for data and twiddle factor in the device memory
	gpuErrchk(cudaMalloc((void**)&d_a, size));

	//allocating space for data and twiddle factor in the host memory
	h_a = (Cplx*)malloc(size);
	h_w = (Cplx*)malloc(size / 2);

	//initialize the data and the twiddle factor
	myCos(h_a, N);
	twiddleFactor(h_w, N);

	//data transfer form host to device for data and the twiddle factor
	gpuErrchk(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(d_w, h_w, size / 2, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpyToSymbol(twiddle, h_w, size / 2));

	// number of threads per block
	unsigned int BLOCK_SIZE = 1024;//512;
	// number of block per grid
	unsigned int GRID_SIZE = (N / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;//1;//

	dim3 threads(BLOCK_SIZE, 1, 1);
	dim3 blocks(GRID_SIZE, 1, 1);

	// launching the kernel
	naive_fft << <blocks, threads >> > (d_a, n, N);
	gpuErrchk(cudaPeekAtLastError());

	//gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost));

	//reverse the order of the outputed data
	bitReverse2natural(h_a, N, n);

	// check if any error happened
	err_check(h_a, N);
	//echo(h_a, N);
	printf("COMPLITED SUCCESSFULLY\n");

	// freeing the memory for both the host and device
	free(h_a); free(h_w);
	cudaFree(d_a);

	return 0;
}



__global__ void naive_fft(Cplx* a, int n, int size) {

	// calculating the global position of thread in the grid
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	Cplx tmp[2];

	if (j < size / 2) {
		// calculating the indexes 
		int x0 = j;
		int x1 = x0 + size / 2;

		// each thread perform this loop log2(size) time (number of stages)
		for (int i = 1; i <= n; i++) {
			// calculating the addition part 
			tmp[0].x = a[x0].x + a[x1].x;// real part
			tmp[0].y = a[x0].y + a[x1].y;// img part

			// calculating the subtraction part
			tmp[1].x = a[x0].x - a[x1].x;// real part
			tmp[1].y = a[x0].y - a[x1].y;// img part

			// calculating the index for the twiddle factor
			int r = (j >> (i - 1)) * (1 << (i - 1));

			//synchonize all threads
			__syncthreads();

			// putting stage data back to global memory 
			a[2*x0] = tmp[0];

			a[2*x0 + 1].x = (tmp[1].x * twiddle[r].x) - (tmp[1].y * twiddle[r].y);
			a[2*x0 + 1].y = (tmp[1].x * twiddle[r].y) + (tmp[1].y * twiddle[r].x);

			// wait for all thread to finish writing to global memory
			__syncthreads();
		}
	}
}

//function to calculate the twiddle factor
void twiddleFactor(Cplx* w, int size) {
	for (int i = 0; i < (size / 2); i++) {
		w[i].x = (float)cos(i * 2 * PI / size);
		w[i].y = (float)sin(-i * 2 * PI / size);
	}
}

// example to initialize data
void myCos(Cplx* a, int size) {
	int Fs = 1000;
	int f = 60;
	for (int i = 0; i < size; i++) {
		float x = (float)(i * 2 * PI * f / Fs);
		a[i].x = cos(x);
		a[i].y = 0;
	}
}

// function to print the data
void echo(Cplx* a, int size) {
	for (int i = 0; i < size; i++) {
		printf("A[%d] = %f --- A[%d] = %f \n", i, a[i].x, i, a[i].y);
	}
}

// error checking function comparing the result of my FFT with cuFFT
void err_check(Cplx* b, int size) {
	unsigned int BATCH = 1;
	unsigned int NX = N;

	cufftComplex* data = (cufftComplex*)malloc(NX * sizeof(cufftComplex));
	cufftComplex* output = (cufftComplex*)malloc(NX * sizeof(cufftComplex));

	cufftComplex* d_data;
	cufftComplex* d_output;

	cudaMalloc((void**)&d_data, NX * sizeof(cufftComplex));
	cudaMalloc((void**)&d_output, NX * sizeof(cufftComplex));

	cumyCos(data, size);

	cudaMemcpy(d_data, data, NX * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	cufftHandle plan;

	cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);

	cufftExecC2C(plan, d_data, d_output, CUFFT_FORWARD);

	cudaMemcpy(output, d_output, NX * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		if (abs(output[i].x - b[i].x) > 0.1) {
			printf("NOT EQUATL 1\n");
		}
		if (abs(output[i].y - b[i].y) > 0.1) {
			printf("NOT EQUATL 2\n");
		}
	}

	cufftDestroy(plan);
	cudaFree(d_data);
	cudaFree(d_output);

	free(data);
	free(output);
}


unsigned int bitReverse(unsigned int num, unsigned int bit_num) {
	unsigned int count = sizeof(unsigned int) * 8 - 1;
	unsigned int reverse_num = num;

	num >>= 1;
	while (num)
	{
		reverse_num <<= 1;
		reverse_num |= num & 1;
		num >>= 1;
		count--;
	}
	reverse_num <<= count;
	reverse_num >>= (sizeof(unsigned int) * 8 - bit_num);

	return reverse_num;
}

void swap(float* a, float* b) {
	float tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
}

void bitReverse2natural(Cplx* a, unsigned int size, unsigned int n) {
	for (unsigned int i = 0; i < size; i++) {
		unsigned int j = bitReverse(i, n);
		if (i < j) {
			swap(&a[i].x, &a[j].x);
			swap(&a[i].y, &a[j].y);
		}
	}
}

// function to initialize the data for cuFFT function
// same data as my FFT to compare the result
void cumyCos(cufftComplex* a, int size) {
	int Fs = 1000;
	int f = 60;
	for (int i = 0; i < size; i++) {
		float x = (float)(i * 2 * PI * f / Fs);
		a[i].x = (float)cos(x);
		a[i].y = 0.0;
	}
}


