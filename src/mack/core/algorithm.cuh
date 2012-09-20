#ifndef ALGORITHM_H_
#define ALGORITHM_H_

#include <cuda_runtime.h>


namespace mack {

/**
 * @namespace mack::core
 * @brief Contains core elements of mack
 *
 * @see MAX_CAND_SIZE
 */
namespace core {

/**
 * @brief Sets to size for every Candidate.
 *
 * @details This must be a fixed value because dynamic allocating in the cuda kernel is
 * very slow and copying fixed arrays to the graphic card is the way to go. Be carefull: changing this value can greatly affect
 * all of macks performance and inner workings.
 */
#define MAX_CAND_SIZE 20

/**
 * @brief Prepares and launches to Kernel for the cracking process
 * @param callback callback to be used for the found targets
 * @param target_loader target_loader that is used for loading the targets
 */
struct candidate{
	unsigned char value[MAX_CAND_SIZE];
	size_t length;
	void init(){
		memset(value, 0, MAX_CAND_SIZE);
		length = 0;
	}
	void init(size_t message_length){
		memset(value, 0, MAX_CAND_SIZE);
		length = message_length;
	}
};

/**
 * @option_type{algorithms,Algorithms}
 * @option_type_class{mack::core::Algorithm}
 */
/**
 * @class Algorithm
 * @brief Defines a cryptographic algorithm, e.g. a cipher or a hash function.
 *
 * @details If you implement yout own algorithm please derive from this base class.
 * The implementation should processes a candidate and returns its hash/ciphertext.
 *
 * @date 13.09.2012
 * @version 0.1
 */
class Algorithm {
public:
//	__device__ Algorithm(){;}


	/**
	 * @brief Process a candidate and return its corresponding hash/ciphertext
	 * @param key_candidate candidate that will be processed
	 * @param result resulting hash/ciphertext will be written to this array
	 */
	__device__ __host__ 
	virtual void compute_target(mack::core::candidate key_candidate, unsigned char* result) const = 0;

	/**
	 * @brief Returns the length of the expected results in bytes (e.g. the size of the hash function output)
	 * @param key_candidate candidate that will be processed
	 * @param result resulting hash/ciphertext will be written to this array
	 */

	__device__ __host__ 
	virtual unsigned int get_target_size(const size_t length) const = 0;
	
	__device__ __host__ 
	virtual unsigned char* get_name() const = 0;

//	__device__ __host__ 
//	virtual static void init();

	__device__ __host__ 
	virtual ~Algorithm() { };

};

}
}

#endif /* ALGORITHM_H_ */
