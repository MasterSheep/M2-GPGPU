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
	__global__ void filterCUDA(uchar *const dev_input, const uint width, const uint height, uchar *const dev_output)
	{

			for(uint id_y = blockIdx.y * blockDim.y + threadIdx.y; id_y < height; id_y += gridDim.y * blockDim.y)
			{
				for(uint id_x = blockIdx.x * blockDim.x + threadIdx.x; id_x < width; id_x += gridDim.x * blockDim.x)
				{

		//	int id_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
		//	int id_y = ((blockIdx.y * blockDim.y) + threadIdx.y);
			int id = (id_x + id_y * width) * 3;

/*
			if((id / 3) > (width * height))
			{
				return;
			}
			*/
				dev_output[id + 0] = static_cast<uchar>(min(255, (int) (dev_input[id + 0] * 0.393 + dev_input[id + 1] * 0.769 + dev_input[id + 2] * 0.189)));
				dev_output[id + 1] = static_cast<uchar>(min(255, (int) (dev_input[id + 0] * 0.349 + dev_input[id + 1] * 0.686 + dev_input[id + 2] * 0.168)));
				dev_output[id + 2] = static_cast<uchar>(min(255, (int) (dev_input[id + 0] * 0.272 + dev_input[id + 1] * 0.534 + dev_input[id + 2] * 0.131)));

			}
		}
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = width * height * 3 * sizeof(uchar);

		////
		chrGPU.start();

		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_output, bytes);

		chrGPU.stop();

		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		////

		// Copy data from host to device (input arrays)
		cudaMemcpy(dev_input, &input[0], bytes, cudaMemcpyHostToDevice);

		//int nb_block = (int) ((width * height)/1024) + 1;
		int thread_size = 32;
		dim3 nb_block = dim3((width / thread_size) + 1,(height / thread_size) + 1,1);
		dim3 nb_thread = dim3(thread_size, thread_size, 1);

		// Launch kernel
		filterCUDA<<<nb_block, nb_thread>>>(dev_input, width, height, dev_output);

		// Copy data from device to host (output array)
		cudaMemcpy(&output[0], dev_output, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
