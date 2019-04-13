#include <bits/stdc++.h>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ctime>
using namespace std;
__global__ void addWithCuda(double *a, double *b, int size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		a[i] += b[i];
	}
}
__global__ void mulWithCuda(double *a, double *b, int size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		a[i] *= b[i];
	}
}
__global__ void divideSumWithCuda(double *a, int size, int div) {
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	if (i%div == 0) {
		a[i - 1] += a[i - 1 - div / 2];
	}
}
void gpuMul(double *a, double *b, int size, cudaDeviceProp device) {
	double *d_a, *d_b;
	cudaMalloc((void**)&d_a, size * sizeof(double));
	cudaMalloc((void**)&d_b, size * sizeof(double));
	cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
	mulWithCuda << <size / device.maxThreadsPerBlock + 1, device.maxThreadsPerBlock >> > (d_a, d_b, size);
	cudaMemcpy(a, d_a, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
}
double gpuDot(double *a, double *b, int size, cudaDeviceProp device) {
	double *d_a, *d_b;
	cudaMalloc((void**)&d_a, size * sizeof(double));
	cudaMalloc((void**)&d_b, size * sizeof(double));
	cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
	mulWithCuda << <size / device.maxThreadsPerBlock + 1, device.maxThreadsPerBlock >> > (d_a, d_b, size);
	long long x = 2;
	while (x <= size) {
		divideSumWithCuda << <size / device.maxThreadsPerBlock + 1, device.maxThreadsPerBlock >> > (d_a, size, x);
		x *= 2;
	}
	double fans = 0;
	double ans;
	int temp = size;
	while (temp) {
		if (temp % 2) {
			cudaMemcpy(&ans, &d_a[size - 1], sizeof(double), cudaMemcpyDeviceToHost);
			fans += ans;
			size -= size & (-size);
		}
		temp /= 2;
	}
	cudaFree(d_a);
	cudaFree(d_b);
	return fans;
}
void gpuAdd(double *a, double *b, int size, cudaDeviceProp device) {
	double *d_a, *d_b;
	cudaMalloc((void**)&d_a, size * sizeof(double));
	cudaMalloc((void**)&d_b, size * sizeof(double));
	cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
	addWithCuda << <size / device.maxThreadsPerBlock + 1, device.maxThreadsPerBlock >> > (d_a, d_b, size);
	cudaMemcpy(a, d_a, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
}
