/*
 * rc4_algorithm.hpp
 *
 *  Created on: 20.06.2012
 *      Author: paul
 */

#ifndef RC4_ALGORITHM_HPP_
#define RC4_ALGORITHM_HPP_

#include <mack/core/algorithm.cuh>
#include <boost/program_options.hpp>

namespace mack{

/**
 * @class RC4_Algorithm
 * @is_of_type{algorithms}
 * @brief Implementation of RC4 algorithm.
 *
 * The implementation of the RC4 stream cipher. Inputs are keystreams and outputs are the 
 * corresponding keys.
 * @author Mack the Knife
 * @date 20.06.2012
 * @version 0.1
 */

#define RC4_STATE_LENGTH 256

class RC4_Algorithm : public mack::core::Algorithm {
public:
	__device__ __host__ RC4_Algorithm(){};

	__device__ __host__ void compute_target(mack::core::candidate, unsigned char *) const;

	__device__ __host__ unsigned int get_target_size(const size_t length) const;
	__device__ __host__ unsigned char* get_name() const;

	__device__ __host__ static void init();

	__device__ __host__ ~RC4_Algorithm();
private:
	__device__ __host__ unsigned char advance(unsigned char i, unsigned char j, unsigned char* state) const;
};

}

#endif /* RC4_ALGORITHM_HPP_ */
