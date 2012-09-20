#ifndef BRUTE_CRACKER_H_
#define BRUTE_CRACKER_H_

#include <mack/core/cracker.hpp>
#include <mack/options/values.hpp>
#include <cuda_runtime.h>
#include <mack/core/algorithm.cuh>
#include <mack/target_loader.hpp>

//keytable
#include <mack/keys/keytable.cuh>
#include <mack/keys/key.cuh>

//some helpers
#include "rainbow_cuda_cracker_helper.cuh"

namespace mack{

#define BRUTE_BLOCKS 12
#define BRUTE_THREADS 640

/**
 * @class Brute_Cracker
 * @is_of_type{crackers}
 * @brief This is a brute force cracker.
 *
 * The brute force cracker takes every possible message
 * and checks if this is the right one. Therefore it could
 * take a very long time to do this.
 *
 * @option{m,length} message-, keystream- or keylength, every value up to 20 is possible
 * @option{k,keytable} Choose between 'full_7bit', 'full_8bit',
 * 'visible_ascii' or 'visible_german' charset.
 * @option{d,device,0} Choose the device, default device is 0
 * @option{t,target-file} Choose the file which contains the targets which should be cracked.
 * @template_option{ALGORITHM,a,algorithm,algorithms}
 * Choose the algorithm to be cracked.

 * @author Paul Kramer
 * @date 29.06.2012
 * @version 0.1
 */
template<class ALGORITHM>
class Brute_Cracker : public mack::core::Cracker {
public:
	Brute_Cracker(mack::options::values const* values);
  void crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const;
	~Brute_Cracker();

private:
	const std::string _target_file_path;
	const int _devID;
	const size_t _keylength;
	const std::string _keytable;
};

//Kernel forward declaration
//	template <class ALGORITHM>
//	__global__
//	void brute_kernel(unsigned char* targets, long target_count, bool* targets_found, long total_key_count, Keytable* device_keytable, size_t keylength);

#endif /* BRUTE_CRACKER_H_ */

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

template<class ALGORITHM>
Brute_Cracker<ALGORITHM>::Brute_Cracker(mack::options::values const* values) :
_target_file_path(values->get("target-file")),
_devID(values->cast<int>("device")),
_keylength(values->cast<size_t>("length")),
_keytable(values->get("keytable"))
{
//	ALGORITHM::init(options);
}
/**
 * @brief Prepares some data to improve the performance of the cracker.
 * @details This method fills a given array of keys with new key objects.
 * @param keys a device array of key pointers to fill, one key per thread
 * @param keytable the keytable object, which is important for new key objects
 * @param keylength the length of the keys
 */
template< class ALGORITHM >
__global__
void
prepare_keys_kernel(Key** keys, Keytable* keytable, size_t keylength)
{
	//get unique thread id
	int threadid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;

	//generating new key objects and store the pointers into the array
	keys[threadid] = new Key(keytable, keylength);
	//initialize the keys
	keys[threadid]->increment(threadid);

}

/**
 * @brief This method frees graphics cards memory from data which was generates in prepare_keys_kernel.
 * @see prepare_keys_kernel
 */

template< class ALGORITHM >
__global__
void
clean_keys_kernel(Key** keys)
{
	//get unique thread id
	int threadid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;

	//removing objects
	delete(keys[threadid]);

}

/**
 * @brief Brute force kernel.
 * @details This is the heart of the cracker, the brute force Cuda kernel. It takes some arguments:
 * @param targets the targets to search for as array of the form: target1target2target3,
 * 			using the number of the targets and the known target length on can divide them
 * @param number_of_targets the number of the targets to search for
 * @param keys an array of keys, every thread gets exactly one key
 * @param keylength the length of the keys
 * @param ciphertexts this variable is needed to improve the performance. During every brute force kernel call
 * 			every thread needs some space to store the temporary ciphertexts (or in case of a hash function for the current hash).
 * 			Instead of malloc some memory and free it during every thread, we build one array, which is shared between all threads.
 * @param threadcount we need the number of threads while the operation, therefore we take this variable,
 * 			it is faster to share this instead of recalculating it
 * @param results an array of results. For every target we need one result, therefore we take this array.
 *
 */
template< class ALGORITHM >
__global__
void
brute_kernel(unsigned char* targets, unsigned long number_of_targets,
		Key** keys, unsigned long number_of_keys, size_t keylength,
		unsigned char* ciphertexts,
		unsigned int threadcount, Result* results)
{

	//get unique thread id
	int threadid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
	//break if we are at the end (may be dangerous)
//	if(threadid >= number_of_keys) return;

	//algorithm
	ALGORITHM algorithm;

	unsigned int number_of_keys_per_thread = (unsigned int)(number_of_keys / threadcount)+1l;
	int targetlength = algorithm.get_target_size(keylength);

	//init the current candidate
	mack::core::candidate candidate;
	memset(candidate.value, 0, sizeof(mack::core::candidate));
	candidate.length = keylength;

	//every thread has to calculate multiple keys per kernel call
	for(int j = 0; j < number_of_keys_per_thread; ++j)
	{
		//copy the key as candidate and encipher (or hash) it
		memcpy(candidate.value, keys[threadid]->get_key_string(), keylength);
		algorithm.compute_target(
				candidate,
				(ciphertexts + threadid * targetlength)
				);

		//then search within the targets for a matching one
		for(long i = 0; i < number_of_targets; ++i)
		{
			//if one target hast the same value as the candidate, we found one!
			if(cudaMemCmp(targets + (i * targetlength),
					(ciphertexts + threadid * targetlength),
					targetlength)){
				//store the candidate in result field and set this target as solved,
				//sometimes more than one thread found (different) result(s), then we overwrite it, because it does not matter
				memcpy(results[i].key, candidate.value, keylength);
				results[i].found = true;
			}
		}
		//increment the own key to the next one for future work
		keys[threadid]->increment(threadcount);
	}

}

/**
 * @brief The crack method.
 * @see Cracker::crack
 */
template< class ALGORITHM >
void
Brute_Cracker<ALGORITHM>::crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const
{
	//inti cuda device properties field
	struct cudaDeviceProp prop;

	//init device id
	int devID = _devID;

	//gets some device properties and selects the right device
	cudaSetDevice(devID);
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&prop,devID);

	// Init keylength
	size_t keylength = _keylength;

	// Init keytable filename
	std::stringstream ss;
	//ss << "../src/keys/char_tables/";
	//ss << "../src/keys/char_tables/";
	ss << _keytable;

	// Init chartable
	unsigned int num_chars = 0;
	unsigned char* char_table = char_table_read(ss.str().c_str(), num_chars);

	//init device chartable
	unsigned char* device_char_table;
	cudaMalloc(&device_char_table, num_chars * sizeof(unsigned char));
	cudaMemcpy(device_char_table, char_table, num_chars * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//calc how much keys fit into gpus memory, its the maximum value
	unsigned int number_of_threads = BRUTE_BLOCKS * BRUTE_THREADS;
	if(((float)prop.totalGlobalMem * 0.9) / sizeof(Keytable) < number_of_threads)
	{
		//gpus memory is too small
		std::cout << "ERROR: GPU Memory is too low, please decrease number of blocks or threads."<<std::endl;
		return;
	}

	//init keytable
	Keytable* keytable;
	Keytable* device_keytable = new Keytable(device_char_table, num_chars);
	cudaMalloc((void**)&keytable, sizeof(Keytable));
	cudaMemcpy(keytable, device_keytable, sizeof(Keytable), cudaMemcpyHostToDevice);

	//init keys
	Key** keys;
	cudaMalloc(&keys, number_of_threads * sizeof(Key*));
	cudaMemset(keys, 0, number_of_threads * sizeof(Key*));

	//init algorithm
	ALGORITHM algorithm;
	int targetlength = algorithm.get_target_size(keylength);

	//initialize space for ciphertexts
	unsigned char* ciphertexts;
	cudaMalloc(&ciphertexts, number_of_threads * sizeof(unsigned char) * targetlength);
	cudaMemset(ciphertexts, 0, number_of_threads * sizeof(unsigned char) * targetlength);

	//prepares the keys
	std::cout << "Prepare some information...";

	prepare_keys_kernel < ALGORITHM > <<<BRUTE_BLOCKS, BRUTE_THREADS>>>(keys, keytable, keylength);

	std::cout << "done." << std::endl;

	//load targets
	//init targetloader
	target_loader->init(algorithm.get_target_size(keylength));
	unsigned char* host_targets = target_loader->load_file_8(_target_file_path.c_str());
	unsigned long target_count = target_loader->get_target_count();

	//init device targets and copy the host targets to the device
	unsigned char* device_targets;
	cudaMalloc(&device_targets, sizeof(unsigned char) * target_count * targetlength);
	cudaMemcpy(device_targets, host_targets, sizeof(unsigned char) * target_count * targetlength, cudaMemcpyHostToDevice);

	//redundant code, in normal conditions the cudaMemcpy will do the same, but maybe it prevents from errors
	cudaDeviceSynchronize();

	//calculates the total number of possible keys and the number of keys per percent for the output and the loop
	unsigned long total_number_of_keys = pow(num_chars, keylength);
	std::cout<<"Total number of keys: "<<total_number_of_keys<<std::endl;
	unsigned long number_of_keys_per_percent = ceil(total_number_of_keys / 100);

	//prepare result array
	Result* h_results = (Result*) malloc(target_count * sizeof(Result));
	Result* d_results;
	cudaMalloc(&d_results, target_count * sizeof(Result));
	cudaMemset(d_results, 0,  target_count * sizeof(Result));

	// prepare cuda time measurement, we decided to measure only cuda runtime,
	// because the amount of work for the cpu is not that high and otherwise we will get some mad outputs
	cudaEvent_t start, stop;
	float time = 0.0f;
	float totaltime = 0.0f;
	std::cout << "Start brute force attack!"<<std::endl;
	std::cout << "Number of Keys per Percent: " << number_of_keys_per_percent << std::endl;
	//the main loop, for every percent we search the keys
	for(int percent = 0; percent < 100; ++percent)
	{
		//cuda time measurement
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//calling the brute force kernel
		brute_kernel< ALGORITHM > <<<BRUTE_BLOCKS, BRUTE_THREADS>>>(device_targets, target_count, keys, number_of_keys_per_percent, keylength, ciphertexts, number_of_threads, d_results);
		//stop the time measurement...
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		//..and sync the device, to be sure, that all threads done
		cudaDeviceSynchronize();
		//calculate the runtime and print it on the console
		cudaEventElapsedTime(&time, start, stop);
		std::cout << (percent+1)<< "% done. "<<std::endl;
		std::cout << (number_of_keys_per_percent/time/1000.f) << "M Hashes per Sec."<<std::endl;
		totaltime += time;
		//output the estimated rest time
		std::cout << "Rest: "<< (total_number_of_keys - (percent*number_of_keys_per_percent))/(number_of_keys_per_percent/time*1000.f) << "s"<<std::endl;
		//backcopy results
		cudaMemcpy(h_results, d_results, target_count * sizeof(Result), cudaMemcpyDeviceToHost);
		bool found_all = true;
		for(int i = 0; i < target_count; ++i){
			found_all &= h_results[i].found;
			if(!found_all) break;
		}
		//break the loop, if all targets where found
		if(found_all) break;
	}
//	output all found targets to become an overview
	for(int i = 0; i < target_count; ++i)
	{
		if(h_results[i].found){
//			std::cout << "Found '"<< h_results[i].key<<"': ";
//			printAsHex(host_targets + (targetlength * i), targetlength);
//			std::cout << std::endl;
			mack::core::candidate* cand = new mack::core::candidate();
			cand->init(keylength);
			memcpy(cand->value, h_results[i].key, keylength);
			callback->call(host_targets + (targetlength * i), cand, targetlength);
			delete cand;
		}
	}

	// free the memory
	cudaFree(device_char_table);
	free(host_targets);
	clean_keys_kernel < ALGORITHM > <<<BRUTE_BLOCKS, BRUTE_THREADS>>>(keys);
	cudaFree(keys);


	std::cout << "Done in "<<totaltime / 1000.f<<"s."<<std::endl;
}

template<class ALGORITHM>
Brute_Cracker<ALGORITHM>::~Brute_Cracker() {
}

}//close namespace mack
