/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumCUDA(int *const dev_input_A, int *const dev_input_B, const uint width, const uint height, int *const dev_output)
	{

		int id_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
		int id_y = ((blockIdx.y * blockDim.y) + threadIdx.y);
		int id = id_x * width + id_y;
		if(id > width * height)
		{
			return;
		}

			dev_output[id] = dev_input_A[id] + dev_input_B[id];
	}

	void studentJob(const std::vector<int> &input_A, const std::vector<int> &input_B, const uint width, const uint height, std::vector<int> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		int *dev_input_A = NULL;
		int *dev_input_B = NULL;
		int *dev_output = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = width * height * sizeof(int);

		////
		chrGPU.start();

		cudaMalloc((void **) &dev_input_A, bytes);
		cudaMalloc((void **) &dev_input_B, bytes);
		cudaMalloc((void **) &dev_output, bytes);

		chrGPU.stop();

		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		////

		// Copy data from host to device (input arrays)
		cudaMemcpy(dev_input_A, &input_A[0], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_input_B, &input_B[0], bytes, cudaMemcpyHostToDevice);

		//int nb_block = (int) ((width * height)/1024) + 1;
		int thread_size = 32;
		dim3 nb_block = dim3((height / thread_size) + 1, (width / thread_size) + 1,1);
		dim3 nb_thread = dim3(thread_size, thread_size, 1);

		// Launch kernel
		sumCUDA<<<nb_block, nb_thread>>>(dev_input_A, dev_input_B, width, height, dev_output);

		// Copy data from device to host (output array)
		cudaMemcpy(&output[0], dev_output, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_input_A);
		cudaFree(dev_input_B);
		cudaFree(dev_output);
	}
}
