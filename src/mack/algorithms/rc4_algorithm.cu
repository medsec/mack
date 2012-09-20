/*
 * rc4_algorithm.cpp
 *
 *  Created on: 20.06.2012
 *      Author: paul
 */

#include "rc4_algorithm.cuh"

namespace mack{

//__device__ __host__
//RC4_Algorithm::RC4_Algorithm()
//{}

__device__ __host__
void
RC4_Algorithm::compute_target(mack::core::candidate key_candidate, unsigned char* target_candidate) const
{
	unsigned char state_[RC4_STATE_LENGTH];
	unsigned char i_;
	unsigned char j_;
	//init RC4
	// Initialize permutation
	unsigned int index = 0;
	while (index < RC4_STATE_LENGTH) {
		state_[index] = index;
		++index;
	}

	// Use key
	index = 0;
	unsigned char j = 0;
	unsigned char tmp;

	while (index < RC4_STATE_LENGTH) {
		j = (j + state_[index] + key_candidate.value[index % key_candidate.length]);
		// swap
		tmp = state_[index];
		state_[index] = state_[j];
		state_[j] = tmp;

		++index;
	}

	i_ = 0;
	j_ = 0;
	//generate keystream
	unsigned int i = 0;
	while (i < key_candidate.length) {
		target_candidate[i] = advance(i_, j_, state_);
		++i;
	}
}

__device__ __host__
unsigned int
RC4_Algorithm::get_target_size(const size_t length) const
{
	return length;
}

__device__ __host__
unsigned char*
RC4_Algorithm::get_name() const
{
	return (unsigned char*)"RC4";
}

__device__ __host__
void
RC4_Algorithm::init() { }

// Get the next byte from the cipher
// return: next byte
__device__ __host__
unsigned char
RC4_Algorithm::advance(unsigned char i, unsigned char j, unsigned char* state) const
{
	++i;
	unsigned char tmp = state[i];
	j += tmp;
	state[i] = state[j];
	state[j] = tmp;
	tmp = state[i] + state[j];
	return state[tmp];
}

__device__ __host__
RC4_Algorithm::~RC4_Algorithm(){}

}


