#ifndef RAINBOW_CUDA_CRACKER_HELPER
#define RAINBOW_CUDA_CRACKER_HELPER



#include <cuda_runtime.h>
#include <mack/keys/key.cuh>
#include <sstream>
#include <mack/core/algorithm.cuh>
#include <stdio.h>

//mmap inlcudes
#include <stdlib.h>
//#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace mack{


/**
 * @brief Defines the maximum keylength.
 * @details Do not change this value to a much higher value, because this will decrease the speed and increase the amount of memory which mack needs.
 */
#define MAX_KEYLENGTH 8

/**
 * @brief Struct which represents one line of the rainbow chain.
 * 
 * @details It contains the first and the last plaintext or key with maximum length of MAX_KEYLENGTH.
 * @see MAX_KEYLENGTH
 */
struct Chain{
	unsigned char key_begin[MAX_KEYLENGTH];
	unsigned char key_end[MAX_KEYLENGTH];
	/**
	 * Initializes all members of the struct with 0.
	 */
	void init(){
		memset(key_begin, 0, MAX_KEYLENGTH);
		memset(key_end, 0, MAX_KEYLENGTH);
	}
};
/**
 * @brief Struct for the results.
 * @details It contains the key, plaintext or message boolean to show that the key was found or not and the length of the key.
 * Maximum length for the plaintext is 20 characters.
 * @see MAX_CAND_SIZE
 */
struct Result{
	unsigned char key[MAX_CAND_SIZE];
	bool found;
	size_t length;
};

__device__ __host__ void reduce(Key* key, int wordlength, int factor1);
__device__ int cudaMemCmp(const unsigned char* left, const unsigned char* right, int length);
unsigned int string_to_unsigned_int(std::string);
__host__ __device__ void printAsHex(unsigned char* text, int length);
void get_table_dimensions(Keytable* keytable, 
		unsigned int& chainlength, 
		long long int& tablelength, 
		int keylength);
bool file_exists(const char * filename);

}

#endif 
