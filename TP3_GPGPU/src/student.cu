/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== EX 1
    __global__ void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{

    extern __shared__ uint sharedMemory[];

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idThread = threadIdx.x;

    if(index >= size)
      sharedMemory[idThread] = 0;
    else
      sharedMemory[idThread] = dev_array[index];

    __syncthreads();
    for(int i = 1; i < blockDim.x; i *= 2) {
      if(idThread % (2*i) == 0)
        sharedMemory[idThread] = umax(sharedMemory[idThread], sharedMemory[idThread + i]);

      __syncthreads();
    }

    if(idThread == 0)
       dev_partialMax[blockIdx.x] = sharedMemory[idThread];
	}

	// ==================================================== EX 2
  __global__ void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
{

  extern __shared__ uint sharedMemory[];

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idThread = threadIdx.x;

  if(index >= size)
    sharedMemory[idThread] = 0;
  else
    sharedMemory[idThread] = dev_array[index];

  __syncthreads();
  for(int i = (blockDim.x / 2); i > 0; i /= 2) {
    if(idThread < i)
      sharedMemory[idThread] = umax(sharedMemory[idThread], sharedMemory[idThread + i]);

    __syncthreads();
  }

  if(idThread == 0)
     dev_partialMax[blockIdx.x] = sharedMemory[idThread];
}

// ==================================================== EX 3
__global__ void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
{

extern __shared__ uint sharedMemory[];

int index = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
int idThread = threadIdx.x;

if(index >= size)
  sharedMemory[idThread] = 0;
else {
  if(index + (blockDim.x / 2) < size)
    sharedMemory[idThread] = umax(dev_array[index], dev_array[index + blockDim.x]);
  else
    sharedMemory[idThread] = dev_array[index];
}

__syncthreads();
for(int i = (blockDim.x / 2); i > 0; i /= 2) {
  if(idThread < i)
    sharedMemory[idThread] = umax(sharedMemory[idThread], sharedMemory[idThread + i]);

  __syncthreads();
}

if(idThread == 0)
   dev_partialMax[blockIdx.x] = sharedMemory[idThread];
}

// ==================================================== EX 4
__global__ void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
{

extern __shared__ uint sharedMemory[];

int index = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
int idThread = threadIdx.x;
int i = 0;

if(index >= size)
  sharedMemory[idThread] = 0;
else {
  if(index + (blockDim.x / 2) < size)
    sharedMemory[idThread] = umax(dev_array[index], dev_array[index + blockDim.x]);
  else
    sharedMemory[idThread] = dev_array[index];
}

__syncthreads();
for(i = (blockDim.x / 2); i > 0; i /= 2) {

  if(i < 32)
    break;

  if(idThread < i)
    sharedMemory[idThread] = umax(sharedMemory[idThread], sharedMemory[idThread + i]);

  __syncthreads();
}

volatile uint * vShardMemory = sharedMemory;

if(idThread < i) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + i]);
i /= 2;
if(idThread < i) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + i]);
i /= 2;
if(idThread < i) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + i]);
i /= 2;
if(idThread < i) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + i]);
i /= 2;
if(idThread < i) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + i]);

if(idThread == 0)
   dev_partialMax[blockIdx.x] = sharedMemory[idThread];
}

// ==================================================== EX 5
template <typename T> T TMax(T x, T y)
{
    return (x > y) ? x : y;
}

template <unsigned int N>
__global__ void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax)
{

extern __shared__ uint sharedMemory[];

int index = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
int idThread = threadIdx.x;

if(index >= size)
  sharedMemory[idThread] = 0;
else {
  if(index + (blockDim.x / 2) < size)
    sharedMemory[idThread] = umax(dev_array[index], dev_array[index + blockDim.x]);
  else
    sharedMemory[idThread] = dev_array[index];
}

volatile uint * vShardMemory = sharedMemory;

__syncthreads();

if(N >= 1024) {
  if(idThread < 512) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 512]);
  __syncthreads();
}

if(N >= 512) {
  if(idThread < 256) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 256]);
  __syncthreads();
}

if(N >= 256) {
  if(idThread < 128) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 128]);
  __syncthreads();
}

if(N >= 128) {
  if(idThread < 64) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 64]);
  __syncthreads();
}
if(N >= 64) {
  if(idThread < 32) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 32]);
  __syncthreads();
}

if(idThread < 16) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 16]);
if(idThread < 8) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 8]);
if(idThread < 4) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 4]);
if(idThread < 2) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 2]);
if(idThread < 1) vShardMemory[idThread] = umax(vShardMemory[idThread], vShardMemory[idThread + 1]);

if(idThread == 0) dev_partialMax[blockIdx.x] = sharedMemory[idThread];
}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);

        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);

        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);

        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);

        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);

        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
