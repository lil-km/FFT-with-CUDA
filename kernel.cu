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

#define PI 3.14159265358979323846
#define N  8192//2048

__global__ void naive_fft(float* a, float* t, int n, int size);
void err_check(float *b, int size);
void twiddleFactor(float* w, int size);
void myCos(float* a, int size);
void echo(float* a, int size);
void bitReverse2natural(float* a, unsigned int size, unsigned int n);
void swap(float* a, float* b);
unsigned int bitReverse(unsigned int num, unsigned int bit_num);
void cumyCos(cufftComplex* a, int size);


int main(void) {
	// host data pointer
	float* h_a, * h_w;
	// device data pointer
	float* d_a, * d_w;

	// size in byte for data, multiplying by factor of two for the real and imaginary part
	unsigned int size = 2 * N * sizeof(float);
	//number of iteration to perform the FFT
	unsigned int n = (unsigned int)(log(N) / log(2));

	//allocating space for data and twiddle factor in the device memory
	gpuErrchk(cudaMalloc((void**)&d_a, size));
	gpuErrchk(cudaMalloc((void**)&d_w, size / 2));
		
	//allocating space for data and twiddle factor in the host memory
	h_a = (float*)malloc(size);
	h_w = (float*)malloc(size / 2);

	//initialize the data and the twiddle factor
	myCos(h_a, N);
	twiddleFactor(h_w, N);

	//data transfer form host to device for data and the twiddle factor
	gpuErrchk(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_w, h_w, size / 2, cudaMemcpyHostToDevice));

	// number of threads per block
	unsigned int BLOCK_SIZE = 1024;//512;
	// number of block per grid
	unsigned int GRID_SIZE = (N/2 + BLOCK_SIZE - 1) / BLOCK_SIZE;//1;//

	dim3 threads(BLOCK_SIZE,1,1);
	dim3 blocks(GRID_SIZE,1,1);

	// launching the kernel
	naive_fft<<<blocks, threads>>>(d_a, d_w, n, N);
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
	cudaFree(d_a); cudaFree(d_w);

	return 0;
}

__global__ void naive_fft(float* a, float* t, int n, int size) {

	// calculating the global position of thread in the grid
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	float tmp[4];

	if(j < size/2){
		// calculating the indexes 
		int x0 = 2 * j;
		int x1 = 2 * j + 1;
		int x2 = x0 + size;
		int x3 = x1 + size;
		int x4 = 4 * j;

		// each thread perform this loop log2(size) time (number of epoch)
		for (int i = 1; i <= n; i++) {
			// calculating addition part 
			tmp[0] = a[x0] + a[x2];// real part
			tmp[1] = a[x1] + a[x3];// img part

			// calculating subtraction part
			tmp[2] = a[x0] - a[x2];// real part
			tmp[3] = a[x1] - a[x3];// img part

			// calculating the index for the twiddle factor
			int r = (j >> (i - 1)) * (1 << (i - 1));

			//synchonize all threads
			__syncthreads();

			// putting data back to global memory 
			a[x4] = tmp[0];
			a[x4 + 1] = tmp[1];

			a[x4 + 2] = (tmp[2] * t[2 * r]) - (tmp[3] * t[2 * r + 1]);
			a[x4 + 3] = (tmp[2] * t[2 * r + 1]) + (tmp[3] * t[2 * r]);

			// wait for all thread to finish writing to global memory
			__syncthreads();
		}
	}
}

//function to calculate the twiddle factor
void twiddleFactor(float* w, int size) {
	for (int i = 0; i < (size / 2); i++) {
		w[2 * i] = (float)cos(i * 2 * PI / size);
		w[2 * i + 1] = (float)sin(-i * 2 * PI / size);
	}
}

// example to initialize data
void myCos(float* a, int size) {
	int Fs = 1000;
	int f = 60;
	for (int i = 0; i < size; i++) {
		float x = (float)(i * 2 * PI * f / Fs);
		a[2 * i] = cos(x);
		a[2 * i + 1] = 0;
	}
}

// function to print the data
void echo(float* a, int size) {
	for (int i = 0; i < size; i++) {
		printf("A[%d] = %f - A[%d] = %f \n", i, a[2 * i], i, a[2 * i + 1]);
	}
}

// error checking function comparing the result of my FFT with cuFFT
void err_check(float *b, int size) {
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
		if (abs(output[i].x - b[2 * i]) > 0.1) {
			printf("NOT EQUATL 1\n");
		}
		if (abs(output[i].y - b[2 * i + 1]) > 0.1) {
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

void bitReverse2natural(float* a, unsigned int size, unsigned int n) {
	for (unsigned int i = 0; i < size; i++) {
		unsigned int j = bitReverse(i, n);
		if (i < j) {
			swap(&a[2 * i], &a[2 * j]);
			swap(&a[2 * i + 1], &a[2 * j + 1]);
		}
	}
}

// function to print the data for cuFFT function
void cumyCos(cufftComplex* a, int size) {
	int Fs = 1000;
	int f = 60;
	for (int i = 0; i < size; i++) {
		float x = (float)(i * 2 * PI * f / Fs);
		a[i].x = (float)cos(x);
		a[i].y = 0.0;
	}
}


