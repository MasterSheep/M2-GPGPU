/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 4: J’aime les maths (et j’additionne des matrices)
*
* File: student.hpp
* Author: Jolan OSTER
*/


#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"

namespace IMAC
{
	// Kernel:
	/// TODO

	// - input: input image RGB
	// - output: output image RGB
    void studentJob(const std::vector<int> &input_A, const std::vector<int> &input_B, const uint width, const uint height, std::vector<int> &output);

}

#endif
