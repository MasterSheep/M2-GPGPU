/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idb = blockIdx.x;
		int idt = threadIdx.x;
		int id = idb * blockDim.x + idt;
		if(id < n) {
			dev_res[id] = dev_a[id] + dev_b[id];
		}
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): "
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();

		/// TODO
		cudaMalloc((void **) &dev_a, bytes);
		cudaMalloc((void **) &dev_b, bytes);
		cudaMalloc((void **) &dev_res, bytes);
		///

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays)
		/// TODO
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
		///

		int nb_block = (int) (size/1024) + 1;

		// Launch kernel
		/// TODO
		sumArraysCUDA<<< nb_block, 1024>>>(size, dev_a, dev_b, dev_res);
		///

		// Copy data from device to host (output array)
		/// TODO
		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);
		///

		// Free arrays on device
		/// TODO
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
		///
	}
}
