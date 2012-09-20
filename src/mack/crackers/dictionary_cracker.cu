#ifndef DICT_CRACKER_H_
#define DICT_CRACKER_H_

#include <mack/core/cracker.hpp>
#include <mack/options/values.hpp>
#include <string>
#include <boost/program_options.hpp>
#include <mack/target_loader.hpp>
#include <cuda_runtime.h>
#include <mack/core/algorithm.cuh>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>
#include <mack/crackers/rainbow_cuda_cracker_helper.cuh>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>

namespace mack{

/**
 * @class Dictionary_Cracker
 * @is_of_type{crackers}
 * @brief Cracks targets with a specified dictionary
 *
 * This Cracker tries to crack its targets by hashing every word in a given dictionary.
 * To be more efficient it will read all words into memory until a specified maximum (default 2GB).
 * The kernel that does the calculation on the gpu will then be called as many times as needed until those
 * words cached in the RAM are processed.
 * Using more RAM (with the option -m) should increase the performance of this cracker.
 * More gpu memory will also increase the performance by lowering the number of kernel calls.
 *
 * There are also lots of parameters to make the dictionary more useful.
 * You can alter the case of the first character by using -x. Furthermore you can append a number of
 * 1-8  digits at the end of each word with -m. To suppress some of the output information you can use -q.
 * @option{D,dictionary} A dictionary
 * @option{n,digits,0} Append a n-digit number at the end of each word(0 < n < 9)
 * @option{m,hostmem,2048} max memory used on the host-side in MB (default 2048)
 * @option_switch{q,quiet} switches into quiet mode
 * @option_switch{x,varycase} varies first character case
 * @option{t,target-file} Choose the file which contains the targets which should be cracked.
 * @option{d,device,0} Choose the device, default device is 0
 * @template_option{ALGORITHM,a,algorithm,algorithms}
 * Choose the algorithm to be cracked.
 *
 * @author Felix Trojan
 * @date 17.06.2012
 * @version 0.1
 */

template<class ALGORITHM>
class Dictionary_Cracker : public mack::core::Cracker {
public:
	Dictionary_Cracker(mack::options::values const* values);
    mack::core::candidate* allocate_buffer_from_file(size_t height, size_t offset, size_t& last_read_offset, size_t& file_size, char* memblock, long int& reserved_memblock_size) const;
    unsigned char* allocate_numbers(unsigned int digits, unsigned char* number_list) const;
    void crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const;
	~Dictionary_Cracker();

private:
	const std::string _dict_path;
	const std::string _target_file_path;
    unsigned long _max_memory_usage_host; //Bytes
    unsigned long _max_memory_usage_gpu; //Bytes
    bool _quiet;
    bool _varycase;
    unsigned int _digits;
    unsigned int _max_num;
    const int _devID;
};

template<class ALGORITHM>
__global__ void dict_kernel(mack::core::candidate* key_candidates, unsigned char* device_results, unsigned char* targets, size_t target_count, bool varycase, unsigned int digits, unsigned int max_number, unsigned char* numbers, Result* results);



#endif /* DICT_CRACKER_H_ */

#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Constructor
 * @param values the parameters from the command line
 */
template<class ALGORITHM>
Dictionary_Cracker<ALGORITHM>::Dictionary_Cracker(mack::options::values const* values)
: _dict_path(values->get("dictionary")), _target_file_path(values->get("target-file")),
  _max_memory_usage_host((values->cast<unsigned long>("hostmem"))*1024l*1024l),
  _max_memory_usage_gpu(1l*1024*1024l*1024l),
  _quiet(values->get_boolean("quiet")),
  _varycase(values->get_boolean("varycase")),
  _digits(values->cast<unsigned int>("digits")),
  _max_num(pow(10,_digits) - 1),
  _devID(values->cast<int>("device"))
{
  if(!_quiet){
	  printf("Dictionary Path: %s\n",_dict_path.c_str());
	  printf("Target Path: %s\n", _target_file_path.c_str());
  }

  //init cuda device properties field
  struct cudaDeviceProp prop;

  //temporary device ID
  int devID = _devID;

  //digits not bigger than 8
  if(_digits > 8)
	  _digits = 8;

  //gets some device properties and selects the right device
  cudaSetDevice(devID);
  cudaGetDevice(&devID);
  cudaGetDeviceProperties(&prop,devID);
  _max_memory_usage_gpu = (unsigned long)(0.9f * (float)prop.totalGlobalMem);
}

/**
 * @brief Reads a dictionary into a newly allocated buffer
 * @param height number of candidates that will be in the returned buffer
 * @param offset offset in bytes where the dictionary will be read
 * @param last_read_offset after reading this value will contain the last read position
 * @param file_size file size of the dictionary will be written to this reference
 * @returns Buffer of Candidates that where read from the dictionary
 */
template<class ALGORITHM>
mack::core::candidate*
Dictionary_Cracker<ALGORITHM>::allocate_buffer_from_file(size_t height, size_t offset, size_t& last_read_offset, size_t& file_size,
		char* memblock, long int& reserved_memblock_size) const
{

	//open file at the end to determine file size
	std::ifstream file;
	file.open(_dict_path.c_str(), std::ios::in | std::ios::ate);

	//buffer size
    size_t buf_size = height*sizeof(mack::core::candidate);
    //file size
    file_size = file.tellg();

    //create and initialize candidate buffer
	mack::core::candidate* buffer = new mack::core::candidate[height];
	memset(buffer,0, buf_size);

	//set new buffer size if the rest of file is smaller
    if(file_size < (offset + buf_size - reserved_memblock_size))
    	buf_size = file_size - offset + reserved_memblock_size;

    if(!_quiet)
    	std::cout << "fsize: " << file_size << " bsize: " << buf_size << " offset: " << offset << " height: " << height << "\n\n";

	//reset file flags and jump to the beginning of the file and read a chunk of data (as memblock)
	file.clear();
	file.seekg(offset, std::ios::beg);
	if(offset < file_size)
	{
		file.read(&memblock[reserved_memblock_size], buf_size - reserved_memblock_size);	//file.read(memblock, buf_size);
		//determine new offset
		last_read_offset = offset + buf_size - reserved_memblock_size;
	}

	//locate and copy the words from memblock in the buffer
	//make sure everything in the file is only read once
	unsigned int j = 0;
	unsigned int k = 0;
	char* pch = strchr(memblock,'\n');
	while(k < height && j < buf_size)
	{
		unsigned int pos = pch - memblock;
		memcpy(buffer[k].value, &memblock[j], (pos-j));
		buffer[k].length = (pos-j);
		//printf ("found at %u %s size: %u j: %u\n",pos, buffer[k].v, buffer[k].l, j);
		j = pos+1;
		++k;
		pch=(char*)memchr(pch+1,'\n',buf_size-j);
		//if \n not found set phc to eof
		if(pch == NULL) pch = &memblock[buf_size];
	}
	//save some time by reusing the unhandled data
	//locate memory that has already been read but does not fit into buffer
	reserved_memblock_size = (long int)buf_size - (long int)j;
//	std::cout << "reserved mem block size: "<<reserved_memblock_size<<std::endl;
//	std::cout << "buf_size: "<< buf_size<<std::endl;
//	std::cout << "j: "<<j<<std::endl;
	if((reserved_memblock_size) < 0) reserved_memblock_size = 0;

	memmove(memblock, &memblock[j], reserved_memblock_size);
	memset(&memblock[reserved_memblock_size], 0, j);

	file.close();

	return buffer;
}

/**
 * @brief Dictionary Cracker Cuda Kernel
 * @param key_candidates buffer of candidates that will be hashed
 * @param device_results the hashed candidates
 * @param the targets that have to be compared
 * @param target_count number of targets
 * @param varycase if true the first character of each candidate will be alternated
 * @param digits number of digits append to each candidate
 * @param max_number maximum value of the append number
 * @param numbers array of precomputed numbers to append
 * @param results contains only the found targets
 */
template<class ALGORITHM>
__global__ void
dict_kernel(mack::core::candidate* key_candidates, unsigned char* device_results, unsigned char* targets, size_t target_count, bool varycase, unsigned int digits, unsigned int max_number, unsigned char* numbers, Result* results){
	ALGORITHM alg;

	//compute thread index (one thread per candidate)
	//if a digit is append each thread computes 10^digit candidates
	//if varycase each thread computes 2 candidates
	//if varycase & digit -> 2^digit*2 candidates per thread
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	size_t target_size = alg.get_target_size(0);


	unsigned char* result = &device_results[idx*target_size];

	if(digits > 0)
	{
		mack::core::candidate tmp = key_candidates[idx];
		tmp.length += digits;
		if(tmp.length <= digits) tmp.length = digits + 1;
		if(tmp.length > MAX_CAND_SIZE) tmp.length = MAX_CAND_SIZE;
		for (unsigned int k = 0; k < max_number; ++k) {

			memcpy(&tmp.value[tmp.length-digits],&numbers[k*digits],digits);

			alg.compute_target(tmp, result);


//			if(idx == 0){
//			   printf("i %d number in array %c%c\n",k,numbers[k*digits],numbers[k*digits+1]);
//			   printf("Found %s : ",tmp.value);
//			   printf("\n");
//			   for(size_t j = 0; j < target_size; ++j)
//				{
//					printf("%02x", result[j] & 0x000000ff);
//				}
//			   printf("\n\n");
//
//			   printf("CANDy %s \n",key_candidates[idx].value);
//		    }

				for(unsigned int i = 0; i < target_count; ++i)
				{
					int res = 1;
					size_t length = target_size;

					while(res && (length > 0))
					{
						--length;
						res &= (result[length] == targets[length + i*target_size]);
					}
					if(res){
						   printf("Found %s : ",tmp.value);
						   printf("\n");
						   for(size_t j = 0; j < target_size; ++j)
							{
								printf("%02x", result[j] & 0x000000ff);
							}
						   printf("\n\n");

						   memcpy(results[i].key, tmp.value, tmp.length);
						   results[i].found = true;
						   results[i].length = tmp.length;
					}

					if(varycase){

							char c = tmp.value[0];

							//lower case
							if(c > '`' && c < '{')
								tmp.value[0] -= 32;
							//upper case
							else if(c > '@' && c < '[')
								tmp.value[0] += 32;
							else
								return;

							alg.compute_target(tmp, result);

							res = 1;
							length = target_size;

							while(res && (length > 0))
							{
								--length;
								res &= (result[length] == targets[length + i*target_size]);
							}
							if(res){
								   memcpy(results[i].key, tmp.value, tmp.length);
								   results[i].found = true;
								   results[i].length = tmp.length;

								   printf("Found %s : ",tmp.value);
								   printf("\n");
								   for(size_t j = 0; j < target_size; ++j)
									{
										printf("%02x", result[j] & 0x000000ff);
									}
								   printf("\n\n");
							}
						}

				}
		}
	}
	else{
		alg.compute_target(key_candidates[idx], result);

		for(unsigned int i = 0; i < target_count; ++i)
		{
			int res = 1;
			size_t length = target_size;

			while(res && (length > 0))
			{
				--length;
				res &= (result[length] == targets[length + i*target_size]);
			}
			if(res){
				   memcpy(results[i].key, key_candidates[idx].value, key_candidates[idx].length);
				   results[i].found = true;
				   results[i].length = key_candidates[idx].length;

				   printf("Found %s : ",key_candidates[idx].value);
				   printf("\n");
				   for(size_t j = 0; j < target_size; ++j)
					{
						printf("%02x", result[j] & 0x000000ff);
					}
				   printf("\n\n");
			}

			if(varycase){

					char c = key_candidates[idx].value[0];

					//lower case
					if(c > '`' && c < '{')
						key_candidates[idx].value[0] -= 32;
					//upper case
					else if(c > '@' && c < '[')
						key_candidates[idx].value[0] += 32;
					else
						return;

					alg.compute_target(key_candidates[idx], result);

					res = 1;
					length = target_size;

					while(res && (length > 0))
					{
						--length;
						res &= (result[length] == targets[length + i*target_size]);
					}
					if(res){
						   memcpy(results[i].key, key_candidates[idx].value, key_candidates[idx].length);
						   results[i].found = true;
						   results[i].length = key_candidates[idx].length;

						   printf("Found %s : ",key_candidates[idx].value);
						   printf("\n");
						   for(size_t j = 0; j < target_size; ++j)
							{
								printf("%02x", result[j] & 0x000000ff);
							}
						   printf("\n\n");
					}
				}

		}
	}

}

/**
 * @brief creates array of precomputed numbers to append to each word
 * @param digits number of digits for each number
 * @param number_list array in which the numbers are stored(must be allocated outside or else a memory leak is produced)
 * @returns array of precomputed numbers (size: 2^digits)
 */
template<class ALGORITHM>
unsigned char*
Dictionary_Cracker<ALGORITHM>::allocate_numbers(unsigned int digits, unsigned char* number_list) const
{
	unsigned int max_number = pow(10,digits);
	std::stringstream ss;

	for (unsigned int i = 0; i < max_number; ++i) {
		ss << std::setw(digits) << std::setfill('0') << i;
	}
	memcpy(number_list, ss.str().c_str(), max_number*digits);
	return (unsigned char*)number_list;
}

/**
 * @brief Prepares and launches to Kernel for the cracking process
 * @param callback callback to be used for the found targets
 * @param target_loader target_loader that is used for loading the targets
 */
template<class ALGORITHM>
void
Dictionary_Cracker<ALGORITHM>::crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const
{
  ALGORITHM algorithm;

  //load targets to calculate the memory sizes
  target_loader->init(algorithm.get_target_size(0));
  unsigned char* host_targets = target_loader->load_file_32(_target_file_path.c_str());
  unsigned long target_count = target_loader->get_target_count();

  if(!_quiet){
	  printf("Max Host Memory used: %lu Bytes\n",_max_memory_usage_host);
	  printf("Max GPU Memory used: %lu Bytes\n",_max_memory_usage_gpu);
  }
  unsigned long single_usage = (algorithm.get_target_size(0)*sizeof(unsigned char) + sizeof(mack::core::candidate));
  unsigned long candidate_count_host = ((_max_memory_usage_host -
											  target_count * algorithm.get_target_size(0)*sizeof(unsigned char) -
											  target_count * sizeof(Result)) /
											  single_usage);
  unsigned long candidate_count_gpu = ((_max_memory_usage_gpu -
											  target_count * algorithm.get_target_size(0)*sizeof(unsigned char) -
											  target_count * sizeof(Result)) /
											  single_usage);
  unsigned long real_usage_gpu = candidate_count_gpu * single_usage;
  if(!_quiet){
	  printf("One Thread needs %lu Bytes Memory\n",single_usage);
	  printf("One Kernel call needs %lu Bytes Memory\n",real_usage_gpu);
  }
  unsigned long kernel_call_count = 1;
  unsigned long offset = 0;
  unsigned long dict_file_size = 0;
  if (_max_memory_usage_host > real_usage_gpu){
	  kernel_call_count = (_max_memory_usage_host/real_usage_gpu);
  }

  if(!_quiet){
	  printf("Kernel will be called %lu times\n",kernel_call_count);
	  printf("One kernel will compute %lu candidates\n",candidate_count_gpu);
	  printf("%lu kernels will compute %lu candidates\n",kernel_call_count,candidate_count_host);
	  printf("Blocks %u will be used per Kernel \n",candidate_count_gpu/512);
  }

   //memblock for caching dictionay from HDD
  char* memblock = new char[candidate_count_host * sizeof(mack::core::candidate)];
  long int reserved_memblock_size = 0;
  //prepare result array
  Result* h_results = (Result*) malloc(target_count * sizeof(Result));
  Result* d_results;
  cudaMalloc(&d_results, target_count * sizeof(Result));
  cudaMemset(d_results, 0,  target_count * sizeof(Result));

  //allocate number_list if needed
  unsigned char* d_number_list;
  unsigned int max_number = pow(10,_digits);
  unsigned char* number_list = (unsigned char*) malloc(max_number*_digits);

  if(_digits > 0){
	  allocate_numbers(_digits, number_list);
	  cudaMalloc(&d_number_list, max_number*_digits);
	  cudaMemset(d_number_list, 0, max_number*_digits);
	  cudaMemcpy(d_number_list, number_list, max_number*_digits, cudaMemcpyHostToDevice);
	  cudaDeviceSynchronize();
  }


  //loop counter and time measurement variables
  cudaEvent_t start, stop;
  float gpu_time = 0.0f;
  float gpu_elapsed = 0.0f;
  float cpu_start = clock();
  float time_elapsed = 0.0f;
  unsigned long j = 0;

  //this loop is called as many times until the dictionary is read
  //it allocates as many candidates in the host memory as possible (but not more than _max_memory_usage_host)
  do{
	  //allocate_buffer_from_file gives us the last position where the file was read
	  //we set this position as the new offset for the next iteration
	  mack::core::candidate* host_key_candidates = allocate_buffer_from_file(candidate_count_host, offset,offset,dict_file_size, memblock, reserved_memblock_size);

	  //this loop is called as many times until all candidates in the host memory are processed by the kernel
	  for(unsigned long i = 0; i < kernel_call_count; ++i){

		  mack::core::candidate* device_key_candidates;
		  unsigned char* device_targets;
		  unsigned char* device_results;

		  cudaMalloc(&device_key_candidates, candidate_count_gpu*sizeof(mack::core::candidate));
		  cudaMalloc(&device_targets, target_count*algorithm.get_target_size(0)*sizeof(unsigned char));
		  cudaMalloc(&device_results, candidate_count_gpu*algorithm.get_target_size(0)*sizeof(unsigned char));

		  //copy targets and candidates
		  cudaMemcpy(device_key_candidates, &host_key_candidates[i*candidate_count_gpu], candidate_count_gpu*sizeof(mack::core::candidate), cudaMemcpyHostToDevice);
		  cudaMemcpy(device_targets, host_targets, target_count*algorithm.get_target_size(0)*sizeof(unsigned char), cudaMemcpyHostToDevice);

		  //start time measurement of the kernel
		  cudaEventCreate(&start);
		  cudaEventCreate(&stop);
		  cudaEventRecord(start, 0);

		  dict_kernel<ALGORITHM><<<candidate_count_gpu/512,512>>>(device_key_candidates, device_results, device_targets, target_count, _varycase, _digits, max_number, d_number_list, d_results);

		  cudaEventRecord(stop, 0);
		  cudaEventSynchronize(stop);
		  cudaDeviceSynchronize();
		  cudaEventElapsedTime(&gpu_time, start, stop);

		  //clean up gpu memory
		  cudaFree(device_key_candidates);
		  cudaFree(device_targets);
		  cudaFree(device_results);

		  //compute elapsed time and hashrate
		  gpu_elapsed += gpu_time;
		  time_elapsed =  (gpu_elapsed/1000.0f + (clock() - cpu_start)/CLOCKS_PER_SEC);
		  float now = time_elapsed;
		  std::cout << "Time: " << ((int)now/3600) % 24 << "h " << ((int)now/60) % 60 << "m " << (int)now % 60 << "s    ";
		  std::cout << (candidate_count_gpu * (i+1) + (j*kernel_call_count*candidate_count_gpu))*max_number*((int)(_varycase)+1)/now/1000.f << " KHashes/s\n";
		  printf("Hashes computed: appr. %u\n", (candidate_count_gpu * (i+1) + (j*kernel_call_count*candidate_count_gpu))*max_number*((int)(_varycase)+1));

		  printf("Candidate: %s\n",host_key_candidates[i*candidate_count_gpu].value);

	  }
	  delete[] host_key_candidates;
	  ++j;
  }while(dict_file_size > offset || reserved_memblock_size != 0);
  printf("-------------------------------\n");
  printf("Dictionary File completely read\n");

  //copy results back and send them to the callback
  cudaMemcpy(h_results, d_results, target_count * sizeof(Result), cudaMemcpyDeviceToHost);
  for(int i = 0; i < target_count; ++i)
  {
	if(h_results[i].found){
		mack::core::candidate* cand = new mack::core::candidate();
		cand->init(h_results[i].length);
		memcpy(cand->value, h_results[i].key, h_results[i].length);
		callback->call(target_loader->get_target(i, host_targets), cand, algorithm.get_target_size(0));
		delete cand;
	}
  }

  //clean up
  cudaFree(d_results);
  if(_digits > 0)
  {
	  cudaFree(d_number_list);
  }
  free(h_results);
  free(number_list);

  /*
  for(unsigned int i = 0; i < 6; ++i)
	  for(unsigned int j = 0; j < tgl.get_target_count(); ++j)
		  algorithm.compute_target(key_candidates[i], tgl.get_target(j,targets));*/
  delete[] memblock;
  delete[] host_targets;
}

/**
 * @brief Destructor
 */
template<class ALGORITHM>
Dictionary_Cracker<ALGORITHM>::~Dictionary_Cracker() { }

}

