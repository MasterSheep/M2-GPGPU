/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/
#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

		__constant__ float dev_matConv[1024];

		texture <uchar4, 1, cudaReadModeElementType> dev_img_1D;
		texture <uchar4, 2, cudaReadModeElementType> dev_img_2D;

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";
    	return os;
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}

// ==================================================
__global__ void conv2D_CUDA(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4* const dev_output)
{

	for(int idY = blockIdx.y * blockDim.y + threadIdx.y; idY < imgHeight; idY += gridDim.y * blockDim.y)
	{
		for(int idX = blockIdx.x * blockDim.x + threadIdx.x; idX < imgWidth; idX += gridDim.x * blockDim.x)
		{

			float3 sum = make_float3(0.f,0.f,0.f);
			int id = idY * imgWidth + idX;
			int i, j, x, y;
			uint idMat;
			uchar4 pixel;

			// Apply convolution
			for (j = 0; j < matSize; ++j )
			{
				for (i = 0; i < matSize; ++i )
				{
					x = min(imgWidth  - 1,  max(0, (idX + i - (int) matSize / 2)  ));
					y = min(imgHeight - 1,  max(0, (idY + j - (int) matSize / 2)  ));

					idMat		= j * matSize + i;
					//idImage	= y * imgWidth + x;

					pixel = tex2D(dev_img_2D, x, y);
					sum.x += (float) pixel.x * dev_matConv[idMat];
					sum.y += (float) pixel.y * dev_matConv[idMat];
					sum.z += (float) pixel.z * dev_matConv[idMat];
				}
			}

			dev_output[id].x = (uchar)min(255.f,  max(0.f, sum.x)  );
			dev_output[id].y = (uchar)min(255.f,  max(0.f, sum.y)  );
			dev_output[id].z = (uchar)min(255.f,  max(0.f, sum.z)  );
			dev_output[id].w = 255;

		}
	}

}


__global__ void conv1D_CUDA(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4* const dev_output)
{

	for(int idY = blockIdx.y * blockDim.y + threadIdx.y; idY < imgHeight; idY += gridDim.y * blockDim.y)
	{
		for(int idX = blockIdx.x * blockDim.x + threadIdx.x; idX < imgWidth; idX += gridDim.x * blockDim.x)
		{

			float3 sum = make_float3(0.f,0.f,0.f);
			int id = idY * imgWidth + idX;
			int i, j, x, y;
			uint idMat, idImage;
			uchar4 pixel;

			// Apply convolution
			for (j = 0; j < matSize; ++j )
			{
				for (i = 0; i < matSize; ++i )
				{
					x = min(imgWidth  - 1,  max(0, (idX + i - (int) matSize / 2)  ));
					y = min(imgHeight - 1,  max(0, (idY + j - (int) matSize / 2)  ));

					idMat		= j * matSize + i;
					idImage	= y * imgWidth + x;

					pixel = tex1Dfetch(dev_img_1D, idImage);
					sum.x += (float) pixel.x * dev_matConv[idMat];
					sum.y += (float) pixel.y * dev_matConv[idMat];
					sum.z += (float) pixel.z * dev_matConv[idMat];
				}
			}

			dev_output[id].x = (uchar)min(255.f,  max(0.f, sum.x)  );
			dev_output[id].y = (uchar)min(255.f,  max(0.f, sum.y)  );
			dev_output[id].z = (uchar)min(255.f,  max(0.f, sum.z)  );
			dev_output[id].w = 255;

		}
	}

}

__global__ void convConstCUDA(	const uchar4* const dev_inputImg, const uint imgWidth, const uint imgHeight,
													      const uint matSize,
																uchar4* const dev_output)
{

	for(int idY = blockIdx.y * blockDim.y + threadIdx.y; idY < imgHeight; idY += gridDim.y * blockDim.y)
	{
		for(int idX = blockIdx.x * blockDim.x + threadIdx.x; idX < imgWidth; idX += gridDim.x * blockDim.x)
		{

			float3 sum = make_float3(0.f,0.f,0.f);
			int id = idY * imgWidth + idX;
			int i, j, x, y;
			uint idMat, idImage;
			uchar4 pixel;

			// Apply convolution
			for (j = 0; j < matSize; ++j )
			{
				for (i = 0; i < matSize; ++i )
				{
					x = min(imgWidth  - 1,  max(0, (idX + i - (int) matSize / 2)  ));
					y = min(imgHeight - 1,  max(0, (idY + j - (int) matSize / 2)  ));

					idMat		= j * matSize + i;
					idImage	= y * imgWidth + x;

					pixel = dev_inputImg[idImage];
					sum.x += (float) pixel.x * dev_matConv[idMat];
					sum.y += (float) pixel.y * dev_matConv[idMat];
					sum.z += (float) pixel.z * dev_matConv[idMat];
				}
			}

			dev_output[id].x = (uchar)min(255.f,  max(0.f, sum.x)  );
			dev_output[id].y = (uchar)min(255.f,  max(0.f, sum.y)  );
			dev_output[id].z = (uchar)min(255.f,  max(0.f, sum.z)  );
			dev_output[id].w = 255;

		}
	}

}

	__global__ void convCUDA(	const uchar4* const dev_inputImg, const uint imgWidth, const uint imgHeight,
														const float* const dev_matConv,  const uint matSize,
														uchar4* const dev_output)
{


	for(int idY = blockIdx.y * blockDim.y + threadIdx.y; idY < imgHeight; idY += gridDim.y * blockDim.y)
	{
		for(int idX = blockIdx.x * blockDim.x + threadIdx.x; idX < imgWidth; idX += gridDim.x * blockDim.x)
		{

			float3 sum = make_float3(0.f,0.f,0.f);
			int id = (idY * imgWidth) + idX;
			int i, j, x, y;
			uint idMat, idImage;
			uchar4 pixel;

			// Apply convolution
			for (j = 0; j < matSize; ++j )
			{
				for (i = 0; i < matSize; ++i )
				{
					x = min(imgWidth  - 1,  max(0, (idX + i - (int) matSize / 2)  ));
					y = min(imgHeight - 1,  max(0, (idY + j - (int) matSize / 2)  ));

					idMat		= j * matSize + i;
					idImage	= y * imgWidth + x;

					pixel = dev_inputImg[idImage];
					sum.x += (float)pixel.x * dev_matConv[idMat];
					sum.y += (float)pixel.y * dev_matConv[idMat];
					sum.z += (float)pixel.z * dev_matConv[idMat];
				}
			}

			dev_output[id].x = (uchar)min(255.f,  max(0.f, sum.x)  );
			dev_output[id].y = (uchar)min(255.f,  max(0.f, sum.y)  );
			dev_output[id].z = (uchar)min(255.f,  max(0.f, sum.z)  );
			dev_output[id].w = 255;
		}
	}
}

  void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		ChronoGPU chrGPU;

		// 2 basic arrays for GPU
		uchar4 *dev_inputImg = NULL;
		uchar4 *dev_output   = NULL;

		//
		const size_t imgBytes = imgWidth * imgHeight * sizeof(uchar4);
		const size_t matBytes = matSize  * matSize   * sizeof(uint);

		// Configure kernel
		const dim3 nb_threads = dim3(32, 32, 1); // 1024
		const dim3 nb_blocks = dim3((imgWidth + nb_threads.x - 1) / nb_threads.x, (imgHeight + nb_threads.y - 1) / nb_threads.y, 1);

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////    Exo 1   /////////////////////////////////////////////////////////
		/*
		// 1 array of matrix for GPU
		float  *dev_matConv	 = NULL;

		// Allocate arrays on device (input, ouput and matrix)
		std::cout 	<< "Allocating input and ouput (2 arrays): "  << ( ( 2 * imgBytes ) >> 20 ) << " MB on Device" << std::endl
		          << "Allocating convolution matrix (1 array): "  << ( ( 2 * matBytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMalloc((void **) &dev_inputImg, imgBytes) );
		HANDLE_ERROR( cudaMalloc((void **) &dev_matConv,  matBytes) );
		HANDLE_ERROR( cudaMalloc((void **) &dev_output,   imgBytes) );
		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays)
		std::cout << "Copy input from host to device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(dev_inputImg, inputImg.data(), imgBytes, cudaMemcpyHostToDevice) );
		HANDLE_ERROR( cudaMemcpy(dev_matConv,  matConv.data(),  matBytes, cudaMemcpyHostToDevice) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Launch kernel
		std::cout << "Convolution filter on GPU (" 	<< nb_blocks.x << "x" << nb_blocks.y << " blocks - " << nb_threads.x << "x" << nb_threads.y << " threads)" << std::endl;
		chrGPU.start();
		convCUDA<<<nb_blocks, nb_threads>>>(dev_inputImg, imgWidth, imgHeight, dev_matConv, matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)
		std::cout << "Copy output from device to host" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(output.data(), dev_output, imgBytes, cudaMemcpyDeviceToHost) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Free arrays on device
		cudaFree(dev_matConv);
		*/
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////    Exo 2   /////////////////////////////////////////////////////////
		/*
		// Allocate arrays on device (input, ouput and matrix)
		std::cout 	<< "Allocating input and ouput (2 arrays): "  << ( ( 2 * imgBytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMalloc((void **) &dev_inputImg, imgBytes) );
		HANDLE_ERROR( cudaMalloc((void **) &dev_output,   imgBytes) );
		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays)
		std::cout << "Copy input from host to device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(dev_inputImg, inputImg.data(), imgBytes, cudaMemcpyHostToDevice) );
		HANDLE_ERROR( cudaMemcpyToSymbol(dev_matConv, matConv.data(), matBytes, 0, cudaMemcpyHostToDevice) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Launch kernel
		std::cout << "Convolution filter with const on GPU (" 	<< nb_blocks.x << "x" << nb_blocks.y << " blocks - " << nb_threads.x << "x" << nb_threads.y << " threads)" << std::endl;
		chrGPU.start();
		convConstCUDA<<<nb_blocks, nb_threads>>>(dev_inputImg, imgWidth, imgHeight, matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)
		std::cout << "Copy output from device to host" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(output.data(), dev_output, imgBytes, cudaMemcpyDeviceToHost) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		*/
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////    Exo 3   /////////////////////////////////////////////////////////
		/*
		// Allocate arrays on device (input, ouput and matrix)
		std::cout 	<< "Allocating input and ouput (2 arrays): "  << ( ( 2 * imgBytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMalloc((void **) &dev_inputImg, imgBytes) );
		HANDLE_ERROR( cudaMalloc((void **) &dev_output,   imgBytes) );
		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays)
		std::cout << "Copy input from host to device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(dev_inputImg, inputImg.data(), imgBytes, cudaMemcpyHostToDevice) );
		HANDLE_ERROR( cudaMemcpyToSymbol(dev_matConv, matConv.data(), matBytes, 0, cudaMemcpyHostToDevice) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Bind texture 1D
		std::cout << "Binding 1D Texture" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaBindTexture( 0, dev_img_1D, dev_inputImg, imgBytes) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Launch kernel
		std::cout << "Convolution filter with 1D texture on GPU (" 	<< nb_blocks.x << "x" << nb_blocks.y << " blocks - " << nb_threads.x << "x" << nb_threads.y << " threads)" << std::endl;
		chrGPU.start();
		conv1D_CUDA<<<nb_blocks, nb_threads>>>(imgWidth, imgHeight, matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)
		std::cout << "Copy output from device to host" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(output.data(), dev_output, imgBytes, cudaMemcpyDeviceToHost) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		*/
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////    Exo 4   /////////////////////////////////////////////////////////

		size_t pitch;
		const size_t widthBytes = imgWidth * sizeof(uchar4);

		// Allocate arrays on device (input, ouput and matrix)
		std::cout 	<< "Allocating input and ouput (2 arrays): "  << ( ( 2 * imgBytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMallocPitch( &dev_inputImg, &pitch, widthBytes, imgHeight) );
		HANDLE_ERROR( cudaMalloc((void **) &dev_output,   imgBytes) );
		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays)
		std::cout << "Copy input from host to device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy2D(dev_inputImg, pitch, inputImg.data(), widthBytes, widthBytes, imgHeight, cudaMemcpyHostToDevice) );
		HANDLE_ERROR( cudaMemcpyToSymbol(dev_matConv, matConv.data(), matBytes, 0, cudaMemcpyHostToDevice) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Bind texture 2D
		std::cout << "Binding 2D Texture" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaBindTexture2D(NULL, dev_img_2D, dev_inputImg, imgHeight, imgWidth,  pitch) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Launch kernel
		std::cout << "Convolution filter with 2D texture on GPU (" 	<< nb_blocks.x << "x" << nb_blocks.y << " blocks - " << nb_threads.x << "x" << nb_threads.y << " threads)" << std::endl;
		chrGPU.start();
		conv2D_CUDA<<<nb_blocks, nb_threads>>>(imgWidth, imgHeight, matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)
		std::cout << "Copy output from device to host" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy(output.data(), dev_output, imgBytes, cudaMemcpyDeviceToHost) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::cout << "Comparison of final results" << std::endl;
		compareImages(resultCPU, output);

		// Free array on device
		std::cout << std::endl << "Free memory on GPU" << std::endl;
		cudaFree(dev_output);
		cudaFree(dev_inputImg);
	}
}
