/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: main.cpp
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cstring>
#include <exception>
#include <algorithm>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg)
	{
		std::cerr	<< "Usage: " << prg << " <L> <H> \n - <L> width dimension of the matrice \n - <H> height dimension of the matrice " << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	// Computes sepia of 'input' and stores result in 'output'
	void matrixCPU(const std::vector<int> &input_A, const std::vector<int> &input_B, const uint width, const uint height, std::vector<int> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for (uint i = 0; i < height; ++i)
		{
			for (uint j = 0; j < width; ++j)
			{
				output[i * width + j] = input_A[i * width + j] + input_B[i * width + j];
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Compare two vectors
	bool compare(const std::vector<int> &a, const std::vector<int> &b)
	{
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			return false;
		}
		for (uint i = 0; i < a.size(); ++i)
		{
			// Floating precision can cause small difference between host and device
			if (std::abs(a[i] - b[i]) > 1)
			{
				std::cout << "Error at index " << i << ": a = " << uint(a[i]) << " - b = " << uint(b[i]) << std::endl;
				return false;
			}
		}
		return true;
	}

	int random()
	{
	    return rand();
	}

	// Main function
	void main(int argc, char **argv)
	{

		// Parse command line
		if (argc != 3)
		{
			std::cerr << "Please two dimension..." << std::endl;
			printUsageAndExit(argv[0]);
		}

		// Get input imag
		uint width  = std::atoi(argv[1]);
		uint height = std::atoi(argv[2]);

		std::cout << "Matrix has " << width << " x " << height << " dimension" << std::endl;

		// Create 2 output images
		std::vector<int> outputCPU(height * width);
		std::vector<int> outputGPU(height * width);

		// Init matrix
		std::vector<int> input_A(height * width);
		std::vector<int> input_B(height * width);
		std::generate(input_A.begin(), input_A.end(), random);
		std::generate(input_B.begin(), input_B.end(), random);

		// Computation on CPU
		matrixCPU(input_A, input_B, width, height, outputCPU);

		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input_A, input_B, width, height, outputGPU);

		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(outputCPU, outputGPU))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
	}
}

int main(int argc, char **argv)
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
