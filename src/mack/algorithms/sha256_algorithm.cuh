/*
 * sha256_algorithm.cuh
 *
 *  Created on: 10.07.2012
 *      Author: azzaroff
 */

#ifndef SHA256_ALGORITHM_CUH_
#define SHA256_ALGORITHM_CUH_

#include <mack/core/algorithm.cuh>
#include <boost/program_options.hpp>

namespace mack{

/**
 * @class sha256_Algorithm
 * @is_of_type{algorithms}
 * @brief The SHA256 Hash Algroithm
 *
 * The detailed description of the sha256 algorithm
 * @author Felix Trojan
 * @date 10.07.2012
 * @version 0.1
 */

#define rol(x,n) ((x << n) | (x >> (32-n)))
#define ror(x,n) ((x >> n) | (x << (32-n)))
#define Ch(x,y,z) ((x & y) ^ ( (~x) & z))
#define Maj(x,y,z) ((x & y) ^ (x & z) ^ (y & z))
#define Sigma0(x) ((ror(x,2))  ^ (ror(x,13)) ^ (ror(x,22)))
#define Sigma1(x) ((ror(x,6))  ^ (ror(x,11)) ^ (ror(x,25)))
#define sigma0(x) ((ror(x,7))  ^ (ror(x,18)) ^(x>>3))
#define sigma1(x) ((ror(x,17)) ^ (ror(x,19)) ^(x>>10))

#ifndef uint32_t
  #define uint32_t unsigned int
#endif

class sha256_Algorithm: public mack::core::Algorithm {
public:
	__device__
	sha256_Algorithm();

	__device__ __host__
	void compute_target(mack::core::candidate key_candidate, unsigned char* result)const;

	__device__ __host__
    unsigned int get_target_size(size_t length) const;

	__device__ __host__
	unsigned char* get_name() const;

    __device__ __host__
	~sha256_Algorithm();

private:
	size_t _target_size;
};

}

#endif /* SHA256_ALGORITHM_CUH_ */
