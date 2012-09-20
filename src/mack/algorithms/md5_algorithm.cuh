#ifndef MD5_ALGORITHM_HPP_
#define MD5_ALGORITHM_HPP_

#include <mack/core/algorithm.cuh>
#include <boost/program_options.hpp>

namespace mack{

/**
 * @class MD5_Algorithm
 * @is_of_type{algorithms}
 * @brief The MD5 Hash Algorithm
 *
 * @author Felix Trojan
 * @date 16.06.2012
 * @version 0.1
 */

class MD5_Algorithm: public mack::core::Algorithm {
public:
	__device__
	MD5_Algorithm();

	__device__ __host__
	void compute_target(mack::core::candidate key_candidate, unsigned char* result)const;

	__device__ __host__
    unsigned int get_target_size(size_t length) const;

	__device__ __host__
    static void init(boost::program_options::variables_map const& options);
	
	__device__ __host__ 
	unsigned char* get_name() const;

    __device__ __host__
	~MD5_Algorithm();

private:
	size_t _target_size;
};

}

#endif /* MD5_ALGORITHM_HPP_ */
