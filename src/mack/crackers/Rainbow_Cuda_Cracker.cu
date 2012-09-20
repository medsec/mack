#ifndef RAINBOW_CUDA_CRACKER_H_
#define RAINBOW_CUDA_CRACKER_H_

#include <mack/core/cracker.hpp>
#include <mack/options/values.hpp>
//math includes
#include <math.h>
//mmap inlcudes
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
//keytable
#include <mack/keys/keytable.cuh>
#include <mack/keys/key.cuh>
//cuda include
#include <cuda_runtime.h>
//candidate include
#include <mack/core/algorithm.cuh>

#include "rainbow_cuda_cracker_helper.cuh"

#include <mack/target_loader.hpp>

namespace mack{

#define RAINBOW_GEN_BLOCKS 64
#define RAINBOW_GEN_THREADS 128
#define RAINBOW_BLOCKS 64
#define RAINBOW_THREADS 128
//#define RAINBOW_DEVICE 0

/**
 * @class Rainbow_Cuda_Cracker
 * @is_of_type{crackers}
 * @brief This is a cuda powered rainbow cracker.
 *
 * This cracker uses 8192 cuda threads and up to 80% of your graphics cards memory.
 *
 * @option{m,length} message-, keystream- or keylength, max. value: 8
 * @option{h,chainlength,32} length of every rainbow chain, value should
 * be between 16 and 1024, 32 is default value
 * @option{k,keytable} Choose between 'full_7bit', 'full_8bit',
 * 'visible_ascii' or 'visible_german' charset.
 * @option{r,rainbowtable} Location of your rainbow tables.
 * @option{d,device,0} Choose the device, default device is 0
 * @option{t,target-file} Choose the file which contains the targets which should be cracked.
 * @template_option{ALGORITHM,a,algorithm,algorithms}
 * Choose the algorithm to be cracked.
 *
 * @author Paul Kramer
 * @date 25.06.2012
 * @version 0.1
 */

template<class ALGORITHM>
class Rainbow_Cuda_Cracker : public mack::core::Cracker {
public:
	Rainbow_Cuda_Cracker(mack::options::values const* values);
	void crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const;
	~Rainbow_Cuda_Cracker();

private:
	const std::string _target_file_path;
	const std::string _rainbow_table_path;
	const int _devID;
	const size_t _keylength;
	const size_t _chainlength;
	const std::string _keytable;

	__host__ __device__ void calculate_rainbow_chain(int row, int chainlength, Key* key, int keylength) const;

};

//Kernel forward declaration
//	template <class ALGORITHM>
//	__global__
//	void recoverRainbowKernelFast(unsigned char* targets, int keylength, int row,
//			long long int segmentsize, Chain* map_entry, unsigned char* d_temp_keys, Keytable* keytable, int targetcount, Result* results);

	template< class ALGORITHM >
	__global__
	void
	RainbowGeneratorKernel(int chainlength, int keylength,
			Chain* rainbow, long long int segment,
			const long long int segmentsize, Keytable* keytable);

	template< class ALGORITHM >
	void generate_tables(int devID, unsigned int keylength, unsigned int chainlength, const char* keytable_filename, const char* rainbowtable_filename);


#endif /* RAINBOW_CUDA_CRACKER_H_ */


/**
 * The calculate_rainbow_chain method calculates the chainlength - row last steps of the chain.
 * Arguments:
 * 	- int row: the current row to start from
 * 	- int chainlength: the rainbow chainlength
 * 	- Key* key: the key, neede for the reduction function
 * 	- size_t keylength: the length of the key
 * 	We need this method as host compatible, because some steps of the rainbowcracker are calculated by cpu to increase the speed.
 */
template <class ALGORITHM>
__host__ __device__
void
Rainbow_Cuda_Cracker<ALGORITHM>::calculate_rainbow_chain(int row, int chainlength, Key* key, int keylength) const
{
	//init all variables we need (an algorithm, a keystream/ciphertext/hash, a candidate)
	ALGORITHM algorithm = ALGORITHM();
	unsigned char* local_keystream = (unsigned char*)malloc(sizeof(unsigned char*) * algorithm.get_target_size(keylength));
	mack::core::candidate candidate;
	memset(&candidate, 0, sizeof(candidate));
	candidate.length = keylength;
	//calculate the rainbow chain from row to chainlength
	for(int recalc = (row); recalc < (chainlength-1); ++recalc){
		//copy the current key/plaintext/message to the candidate
		memcpy(candidate.value, key->get_key_string(), keylength);
		//hash/encipher it
		algorithm.compute_target(candidate, local_keystream);
		//set the result back to the key (aka cut it to the rigth length and translate it into human readable chars)
		key->set_key_string(local_keystream);
		//reduce it
		reduce(key, keylength, (recalc+1));
		//now we have a new key candidate in the key object
	}
	//free local memory
	free(local_keystream);
}

/**
 * @brief Rainbow cracker kernel.
 * @details This is the heart of this cracker: the rainbow table cuda kernel. The kernel needs a lot arguments:
 * @param targets the targets to search for (hashes or cipher texts)
 * @param keylength the key length
 * @param row the current row, rows are decreased in every iteration
 * @param segmentsize this variable holds the number of rainbow chains which fits into graphics cards memory
 * @param map_entry this is a part of the whole rainbow table (holds segment size chains)
 * @param d_temp_keys these are the local keys, one key per thread, this variable is used to increase the performance,
 * 		 otherwise we must malloc this variable every time we call the kernel and free it every time
 * @param keytable a key table object, we need it for local keys, these are locally created, because they fit into the registers and it is faster
 * @param targetcount the number of targets
 * @param results an array of results. For every target we need one result, therefore we take this array.
 * @param false_positives this variable stores the number of false positives and is a shared variable
 */
template<class ALGORITHM>
__global__
void
recoverRainbowKernelFast(unsigned char* targets, int keylength, int row,
		long long int segmentsize, Chain* map_entry,
		unsigned char* d_temp_keys, Keytable* keytable, int targetcount, Result* results, unsigned int* false_positives)
{
	//unique thread ID
	int threadid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
	int threadcount = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

	//prepare Key
	Key temp_key = Key(keytable, keylength);

	//prepare cipher
	ALGORITHM algorithm = ALGORITHM();

	//prepare temporapy key
	Key* line_key = new Key(keytable, keylength);

	// loop all targets
	for(int target = 0; target < targetcount; ++target){

		// do no redundant work
		if(!results[target].found){
			//set the temporary key to the stored one (which is the last one in chain, if we found it)
			temp_key.set_key_string(d_temp_keys+(target * keylength));

			//loop over all chains in current rainbow table segment
			for(int line = threadid; line < segmentsize; line += threadcount){

				// compare the keys
				if(cudaMemCmp(temp_key.get_key_string(), map_entry[line].key_end, keylength)){ //match
					//prepare temporary keystream buffer
					unsigned char* local_target = (unsigned char*) malloc(sizeof(unsigned char) * algorithm.get_target_size(keylength));

					//we found the right line and the right row
					//prepare temporary variables
					//now recalculate the key up to the right column
					line_key->set_key_string(map_entry[line].key_begin);
					//recalc chain until row
					mack::core::candidate candidate;
					memset(&candidate, 0, sizeof(candidate));
					candidate.length = keylength;
					for(int temp_row = 0; temp_row < row; ++temp_row){
						memcpy(candidate.value, line_key->get_key_string(), candidate.length);
						algorithm.compute_target(candidate, local_target);
						line_key->set_key_string(local_target);
						reduce(line_key, keylength, temp_row);
					}

		//			proof of key
					memcpy(candidate.value, line_key->get_key_string(), candidate.length);
					algorithm.compute_target(candidate, local_target);
					//store the key if we found the right one
					if(cudaMemCmp(targets + (target * algorithm.get_target_size(keylength)), local_target, algorithm.get_target_size(keylength))){
						printf("Proof of rainbow key has been successful.\n");
						printf("%s\n", line_key->get_key_string());
						results[target].found = true;
						memcpy(results[target].key, line_key->get_key_string(), keylength);
						free(local_target);
						break;
					}else{
						//count if we had a false positive
						atomicAdd(false_positives, 1);
					}
					//free local storage
					free(local_target);
				}
			}
		}
	}
	//free local storage
	if(line_key) delete(line_key);
}

/**
 * @brief Generates rainbow tables
 * @details If you want to use rainbow tables you have to calculate them first. Therefore you will use this kernel,
 * which generates the table, but needs lot of input:
 * @param chainlength the length of the rainbowchain
 * @param keylength the length of the key
 * @param segmentsize the number of chains, which fit into graphics cards memory
 * @param keytable the char table
 */
template< class ALGORITHM >
__global__
void
RainbowGeneratorKernel(int chainlength, int keylength,
		Chain* rainbow, long long int segment,
		const long long int segmentsize, Keytable* keytable)
{
	//get unique thread id
	int threadid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
	//number of threads
	int threadcount = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

	//generate Key Object
	Key key = Key(keytable, keylength);
	//prepare the cipher
	ALGORITHM algorithm = ALGORITHM();
	//move the key to the current segment
	key.increment((segment * segmentsize)+threadid);


	//allocate a temporary key field
	Key tempkey = Key(keytable, keylength);

	//allocate keystream variable
	unsigned char* keystream = (unsigned char*)malloc(sizeof(unsigned char) * algorithm.get_target_size(keylength));

	//init candidate
	mack::core::candidate candidate;
	memset(&candidate, 0, sizeof(candidate));
	candidate.length = keylength;

	//begin with calculating the lines, one line per thread
	for(unsigned long long int line = threadid; line < (segmentsize); line += threadcount){
		//copy key into temp key
		tempkey.set_key_string((unsigned char*)key.get_key_string());
		//at the end of this loop the key will be incremented

		//write the key into the chain
		memcpy(rainbow[line].key_begin, tempkey.get_key_string(), keylength);

		//begin to calculate the chain
		for(int row = 0; row < chainlength; ++row){
			//encipher/hash current plaintext/message
			memcpy(candidate.value, tempkey.get_key_string(), keylength);
			algorithm.compute_target(candidate, keystream);
			tempkey.set_key_string(keystream);
			reduce(&tempkey, keylength, row);
		}
		//write the last key into the chain
		memcpy(rainbow[line].key_end, tempkey.get_key_string(), keylength);
		//finish this line
		//get the next key(line) to calculate
		key.increment(threadcount);
	}
	//free the storage
	free(keystream);
}


#include <stdlib.h>
#include <stdio.h>

template<class ALGORITHM>
Rainbow_Cuda_Cracker<ALGORITHM>::Rainbow_Cuda_Cracker(mack::options::values const* values):
_target_file_path(values->get("target-file")),
_devID(values->cast<int>("device")),
_keylength(values->cast<size_t>("length")),
_chainlength(values->cast<size_t>("chainlength")),
_keytable(values->get("keytable")),
_rainbow_table_path(values->get("rainbowtable"))
{
}

/**
 * @brief Rainbow crackers crack method.
 * @param callback the callback to handle the results
 * @param target_loader loads the targets
 */
template<class ALGORITHM>
void
Rainbow_Cuda_Cracker<ALGORITHM>::crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const
{
	//init some device structures
	struct cudaDeviceProp prop;
	int devID = _devID;

	//select the device and get its properties
	cudaSetDevice(devID);
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&prop,devID);

	// Init keytable filename
	std::stringstream ss;
	ss << _keytable;

	if(!file_exists(ss.str().c_str())){
		perror("Error opening char table file for reading");
		exit(EXIT_FAILURE);
	}

	// Init keytable
	unsigned int num_chars = 0;
	unsigned char* char_table = char_table_read(ss.str().c_str(), num_chars);

	// Init chainlength
	unsigned int chainlength = _chainlength;

	// Init keylength
	unsigned int keylength = _keylength;
	if(keylength > 8)
	{
		printf("Maximum keylength is 8. You entered %d!\n", keylength);
		return;
	}

	//init the char table
	unsigned char* char_table_dev;
	cudaMalloc((void**) &char_table_dev, num_chars * sizeof(unsigned char));
	cudaMemcpy(char_table_dev, char_table, num_chars * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//init the char table objects for device and host
	Keytable* k_host = new Keytable(char_table, num_chars);
	Keytable* key_device = new Keytable(char_table_dev, num_chars); //loads char table into device mem
	Keytable* k_dev;
	cudaMalloc((void**) &k_dev, sizeof(Keytable));
	cudaMemcpy(k_dev, key_device, sizeof(Keytable), cudaMemcpyHostToDevice);

	//init stop values
	long long int table_length = 0;
	get_table_dimensions(k_host, chainlength, table_length, keylength);

//	printf("chainlength: %u\n", chainlength);

	//init the algorithm
	ALGORITHM algorithm = ALGORITHM();

	//init targetloader
	target_loader->init(algorithm.get_target_size(keylength));
	unsigned char* host_targets = target_loader->load_file_8(_target_file_path.c_str());
	unsigned long target_count = target_loader->get_target_count();

	//calculate the mmap filesize
	unsigned long long int filesize = (table_length * sizeof(Chain));

	std::cout << "Filesize: "<<filesize/1024/1024<<"  MBytes."<<std::endl;
	std::cout << "Target filepath: "<< _target_file_path.c_str() << std::endl;

	//mmap variables
	int fd;

	//open file
	//get the file path
	std::string file;
	std::stringstream ssf;
	ssf << _rainbow_table_path;
	ssf << _keytable.substr(_keytable.find_last_of("/")+1, (_keytable.find_last_of(".") - (_keytable.find_last_of("/")+1)));
	ssf << "_";
	ssf << algorithm.get_name();
	ssf << "_";
	ssf << keylength;
	ssf << "_";
	ssf << chainlength;
	ssf << ".mrt";
	file = ssf.str();
	char* filepath = new char[file.length()];
	strcpy(filepath,ssf.str().c_str());

	std::cout<<std::endl;
	std::cout<<"Rainbowtable Filename: "<<filepath<<std::endl;

	//output an error if the rainbow table not exists
	if(!file_exists(filepath)){
		std::cout << "No rainbow table found, do you want to generate one? (Y/N)" <<std::endl;
		unsigned char in;
		std::cin >> in;
		//ask the user for calculating the table
		if(in == 'Y' || in == 'y'){
			std::cout << "calculating table, please wait."<<std::endl;
			//call the method to calculate the tables
			generate_tables<ALGORITHM>(devID, keylength, chainlength, ss.str().c_str(), filepath);
		}else{
			//if the user do not want to calculate tables, quit the program
			return;
		}
	}

	//open the table
	fd = open(filepath, O_RDONLY);
	if (fd == -1) {
		perror("Error opening file for reading");
		exit(EXIT_FAILURE);
	}

	//map it to a variable
	Chain* map = (Chain*)mmap(0, filesize, PROT_READ, MAP_SHARED, fd, 0);
	if (map == MAP_FAILED) {
		close(fd);
		perror("Error mmapping the file");
		exit(EXIT_FAILURE);
	}

	//subdivide keyspace
	long long int segmentsize = (int)floor(((float)prop.totalGlobalMem * 0.75f)/ sizeof(Chain)); //number of chains which fits into 1GB memory
	if(segmentsize > table_length) segmentsize = table_length;
	long long int number_of_segments = ceil ((double)table_length / (double)segmentsize);
	long long int current_segment_size = segmentsize;

	//get targetlength
	int targetlength = algorithm.get_target_size(keylength);

	//get the number of targets in gpus memory
	int max_targets_in_memory = (int)(((float)prop.totalGlobalMem * 0.05f)/ sizeof(Chain));
	int number_of_target_segments = (int)(target_count / max_targets_in_memory);
	if(number_of_target_segments < 1) number_of_target_segments = 1;
	int number_of_targets_to_copy = 0;

	//init the device targets
	unsigned char* d_targets;
	cudaMalloc(&d_targets, targetlength * sizeof(unsigned char) * target_count);
	cudaMemset(d_targets, 0, targetlength * sizeof(unsigned char) * target_count);

	//init the host and device keys
	Key* h_temp_key = new Key(k_host, keylength);

	//init temporary keys, or: malloc space for temporary keys
	unsigned char* d_temp_keys;
	unsigned char* h_temp_keys;
	cudaMallocHost(&(h_temp_keys), keylength * sizeof(unsigned char) * max_targets_in_memory);
	cudaMalloc(&d_temp_keys, keylength * sizeof(unsigned char) * max_targets_in_memory);
	cudaMemset(d_temp_keys, 0, keylength * sizeof(unsigned char) * max_targets_in_memory);

	//prepare result array
	Result* h_results = (Result*) malloc(max_targets_in_memory * sizeof(Result));
	Result* d_results;
	cudaMalloc(&d_results, max_targets_in_memory * sizeof(Result));
	cudaMemset(d_results, 0,  max_targets_in_memory * sizeof(Result));

	//prepare host memory mmap
	Chain* h_map = (Chain*)malloc(segmentsize * sizeof(Chain));

	//init the false positive device variable
	unsigned int* d_false_positives;
	cudaMalloc(&d_false_positives, sizeof(unsigned int));
	cudaMemset(d_false_positives, 0, sizeof(unsigned int));

	//init the false positive host variable
	unsigned int* h_false_positives = new unsigned int();
	*h_false_positives = 0;

	//init some output and time measurement variables
	int oldprogress = 0;
	clock_t start, stop;
	float est_time = 0.0f;
	start = clock();

	std::cout << "Segmentsize: "<<segmentsize<<std::endl;
	//loop over all segments, because sometimes the whole table does not fit into memory, we have to do the work stepwise
	for(int targetsegment = 0; targetsegment < number_of_target_segments; ++targetsegment)
	{
		//calculate the number of targets for this step, because we have only 5% of graphics cards memory to store them, we have to do this maybe stepwise
		number_of_targets_to_copy = target_count - (targetsegment * number_of_target_segments);
		if(number_of_targets_to_copy > max_targets_in_memory) number_of_targets_to_copy = max_targets_in_memory;
		//copy the next number_of_targets_to_copy targets into the memory
		cudaMemcpy(d_targets, host_targets + (targetsegment * number_of_target_segments), number_of_targets_to_copy * targetlength * sizeof(unsigned char), cudaMemcpyHostToDevice);

		//reset the progress
		oldprogress = 0;

		std::cout<< "\nSearching for "<<number_of_targets_to_copy << " targets."<<std::endl;

		//for all rainbow segments loop
		for(long long int segment = 0; segment < number_of_segments; ++segment)
		{
			//calcs the size of last segment
			current_segment_size = table_length - (segment * segmentsize);
			if(current_segment_size > segmentsize) current_segment_size = segmentsize;

			//copy the rainbowtable to the craphics cards memory
			Chain* d_map;
			cudaMalloc(&d_map, segmentsize * sizeof(Chain));


			std::cout << "Loading Segment "<<segment+1 << " from file."<<std::endl;
//			std::cout << "copy mmap to ram"<<std::endl;
			memcpy(h_map, &(map[segment * segmentsize]), current_segment_size * sizeof(Chain));
//			std::cout<< "copy done"<<std::endl;
//			std::cout << "map[0]: '"<<map[segment * segmentsize].key_begin<<"'"<<std::endl;
			cudaMemcpy(d_map,
					h_map,
					current_segment_size * sizeof(Chain),
					cudaMemcpyHostToDevice);
			std::cout << "done." << std::endl;
			cudaDeviceSynchronize();

			//calc the key for the initial round, its nearly the same as in the calculate_rainbow_chain method
			unsigned char* currenttarget = 0;
			if(segment == 0){ //this should reduce the amount of calculations
				for(int i = 0; i < number_of_targets_to_copy; ++i)
				{
					currenttarget = host_targets + ((targetsegment * max_targets_in_memory * targetlength) + (i*targetlength));
					h_temp_key->set_key_string(currenttarget);
					reduce(h_temp_key, keylength, chainlength-1);
					calculate_rainbow_chain( chainlength-1, chainlength, h_temp_key, keylength);
					memcpy((h_temp_keys) + (i*keylength), h_temp_key->get_key_string(), keylength);
				}
			}
			//copy the result to the device vairables
			cudaMemcpy(d_temp_keys, h_temp_keys, number_of_targets_to_copy * keylength * sizeof(unsigned char), cudaMemcpyHostToDevice);
			//for all rows, recalculate the chain and search for the targets
			for(int row = (chainlength - 1); row >= 0; --row)
			{
				//test, if the chain contains the key
				recoverRainbowKernelFast <ALGORITHM> <<<RAINBOW_BLOCKS, RAINBOW_THREADS>>>(d_targets, keylength,
						row, current_segment_size, d_map, d_temp_keys,k_dev, number_of_targets_to_copy, d_results, d_false_positives);

				//recalc the chain on host, while the cuda cores are in full workload
				if(segment == 0){
					for(int i = 0; i < number_of_targets_to_copy; ++i)
					{
						currenttarget = host_targets + ((targetsegment * max_targets_in_memory * targetlength) + (i*targetlength));
						h_temp_key->set_key_string(currenttarget);
						reduce(h_temp_key, keylength, row-1);
						calculate_rainbow_chain(row-1, chainlength, h_temp_key, keylength);
						if(row>0) memcpy((h_temp_keys) + (i*keylength), h_temp_key->get_key_string(), keylength);
					}
				}
				//copy the next results to the graphics card
				if(row>0)cudaMemcpy(d_temp_keys, h_temp_keys, number_of_targets_to_copy * keylength * sizeof(unsigned char), cudaMemcpyHostToDevice);

				//output the progress to the user
				int progress = 0;
				progress = (float)((segment * segmentsize * chainlength) + (chainlength - row) * current_segment_size) / (float)(table_length * chainlength) * 100.f;
				if(progress > oldprogress){
					oldprogress = progress;
					stop = clock();
					est_time = (100-progress) * (((float)(stop - start)/((float)CLOCKS_PER_SEC)) / progress);
					printf("%d%% done. Est. stop %d s\n", progress,(int)est_time);
					printf("Speed: %fM Hashes/s\n", ((segment * segmentsize * chainlength) + (chainlength - row) * current_segment_size) / ((float)(stop - start) / (float)CLOCKS_PER_SEC) / 1000000.f);

				}
				cudaDeviceSynchronize();
			}
			//free some local storage
			if(d_map)
					cudaFree(d_map);
		}
		//backcopy results
		cudaMemcpy(h_results, d_results, number_of_targets_to_copy * sizeof(Result), cudaMemcpyDeviceToHost);

		//output all found targets
		for(int i = 0; i < number_of_targets_to_copy; ++i)
		{
			if(h_results[i].found){
//				std::cout << "Found '"<< h_results[i].key<<"': ";
//				printAsHex(host_targets + (targetsegment * max_targets_in_memory * targetlength) + (targetlength * i), targetlength);
//				std::cout << std::endl;
				mack::core::candidate* cand = new mack::core::candidate();
				cand->init(keylength);
				memcpy(cand->value, h_results[i].key, keylength);
				callback->call(host_targets + (targetlength * i), cand, targetlength);
				delete cand;
			}
		}
	//end of target loop
	}
	//stop the time measurement on host
	stop = clock();

	//copy the false positives to host and output them
	cudaMemcpy(h_false_positives, d_false_positives, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	std::cout << "False positives: "<<*h_false_positives<<std::endl;
	std::cout << std::endl << std::endl << "Total duration: "<< ((float)(stop - start)/((float)CLOCKS_PER_SEC)) << "s."<<std::endl;

	//free a lot of local storage
	if(d_targets) cudaFree(d_targets);
	if(k_dev) cudaFree(k_dev);
	if(d_temp_keys) cudaFree(d_temp_keys);
	if(h_temp_keys) cudaFreeHost(h_temp_keys);
	if(char_table_dev) cudaFree(char_table_dev);
	if(d_results) cudaFree(d_results);
	if(d_false_positives) cudaFree(d_false_positives);

	// Don't forget to free the mmapped memory
	if (munmap(map, filesize) == -1) {
		perror("Error un-mmapping the file");
	}

	// Un-mmaping doesn't close the file, so we still need to do that.
	close(fd);

}

/**
 * @brief This method precalculates the rainbow tables.
 * @details It uses the graphics card to calculate them and writes it into a file.
 */
template< class ALGORITHM >
void
generate_tables(int devID, unsigned int keylength, unsigned int chainlength, const char* keytable_filename, const char* rainbowtable_filename)
{
	//select device
	struct cudaDeviceProp prop;

	//get device information
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&prop,devID);

	//init char table
	unsigned int num_chars = 0;
	unsigned char* char_table = char_table_read(keytable_filename, num_chars);

	//init device char table
	unsigned char* char_table_dev;
	cudaMalloc((void**) &char_table_dev, num_chars * sizeof(unsigned char));
	cudaMemcpy(char_table_dev, char_table, num_chars * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//prepare host and device keytables
	Keytable* k_host = new Keytable(char_table, num_chars);
	Keytable* key_device = new Keytable(char_table_dev, num_chars); //loads char table into device mem
	Keytable* k_dev;
	cudaMalloc((void**) &k_dev, sizeof(Keytable));
	cudaMemcpy(k_dev, key_device, sizeof(Keytable), cudaMemcpyHostToDevice);

	//init stop values
	long long int table_length = 0;
	get_table_dimensions(k_host, chainlength, table_length, keylength);
	std::cout << "Generating "<< table_length << " chains of length "<< chainlength<<"."<<std::endl;

	//prepare file for results
	int fd; //filename: file_keylength.txt
	unsigned long long int filesize = (table_length * sizeof(Chain));
	printf("Filesize: %f MBytes\n", ((float)filesize)/1048576.f);
	int result;
	Chain* map;  /* mmapped array of chains */

	//open mmaped rainbow table file
	fd = open(rainbowtable_filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
	if (fd == -1) {
		perror("Error opening file for writing");
		exit(EXIT_FAILURE);
	}

	//Stretch the file size to the size of the (mmapped) array of ints
	result = lseek(fd, filesize-1, SEEK_SET);
	if (result == -1) {
		close(fd);
		perror("Error calling lseek() to 'stretch' the file");
		exit(EXIT_FAILURE);
	}

	 /* Something needs to be written at the end of the file to
	 * have the file actually have the new size.
	 * Just writing an empty string at the current file position will do.
	 *
	 * Note:
	 *  - The current position in the file is at the end of the stretched
	 *    file due to the call to lseek().
	 *  - An empty string is actually a single '\0' character, so a zero-byte
	 *    will be written at the last byte of the file.
	 */
	result = write(fd, "", 1);
	if (result != 1) {
		close(fd);
		perror("Error writing last byte of the file");
		exit(EXIT_FAILURE);
	}

	// Now the file is ready to be mmapped.
	map = (Chain*)mmap(0, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (map == MAP_FAILED) {
		close(fd);
		perror("Error mmapping the file");
		exit(EXIT_FAILURE);
	}

	printf("Calculating!\n");
	float calc_time = 0.0;

	//divide the keyspace into small segments, which fit into graphics cards memory
	long long int segmentsize = (int)(((float)prop.totalGlobalMem * 0.9)/ sizeof(Chain)); //number of chains which fits into 1GB memory
	if(segmentsize > table_length) segmentsize = table_length;
	long long int number_of_segments = ceil ((double)table_length / (double)segmentsize);
	long long int current_segment_size = segmentsize;

	//get a local rainbow table variable
	Chain* h_map = (Chain*)malloc(segmentsize * sizeof(Chain));

	//some time measurement

	for(int segment = 0; segment < number_of_segments; ++segment){

		//calcs the size of last segment
		current_segment_size = table_length - (segment * segmentsize);
		if(current_segment_size > segmentsize) current_segment_size = segmentsize;

		//allocate chain
		Chain* host_rain = (Chain*)malloc(current_segment_size * sizeof(Chain));
		Chain* device_rain;
		cudaMalloc(&device_rain, current_segment_size * sizeof(Chain));

		//transferring dataset to device - note that pitch is passed as a parameter
		cudaMemcpy(device_rain, host_rain, current_segment_size * sizeof(Chain), cudaMemcpyHostToDevice);

		//do some time measurement
		cudaEvent_t start, stop;
		float time = 0.0f;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		RainbowGeneratorKernel <ALGORITHM> <<<RAINBOW_GEN_BLOCKS, RAINBOW_GEN_THREADS>>>(chainlength, keylength,
				device_rain, segment, current_segment_size, k_dev);

		//finish time measurement
		cudaEventRecord(stop, 0);
		// syncronize the device
		cudaEventSynchronize(stop);

		//copy the output into the rainbow table file, this is done from graphics cards memory to host memory to file,
		//otherwise it will not work or stop unexpected
		std::cout << "copy to file"<<std::endl;
		// Now write chains to the file as if it were memory (an array of chains).
		long offset = segment*segmentsize; //richtig, keinesfalls ändern!
//		cudaMemcpy(host_rain, device_rain, current_segment_size * sizeof(Chain), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_map, device_rain, current_segment_size * sizeof(Chain), cudaMemcpyDeviceToHost);
		memcpy((map+offset), h_map, current_segment_size * sizeof(Chain));
		std::cout << "done."<<std::endl;

		//syncronize the device, this should be redundant, but maybe it prevents from errors
		cudaDeviceSynchronize();

		//output time
		cudaEventElapsedTime(&time, start, stop);
		calc_time += time;

		//free local storage
		if(device_rain)
			cudaFree(device_rain);

		if(host_rain)
			free(host_rain);

		printf("%d %% done.\n", (int)((float)(segment+1)*100.f / (float)number_of_segments));

	}

	// Don't forget to free the mmapped memory
	if (munmap(map, filesize) == -1) {
		perror("Error un-mmapping the file");
		// Decide here whether to close(fd) and exit() or not. Depends...
	}

	// Un-mmaping doesn't close the file, so we still need to do that.
	close(fd);

	//free all variables
	cudaFree(char_table_dev);
	cudaFree(k_dev);
	free(h_map);

	printf ("\nTime for all kernels: %f s\n", calc_time/1000);
//		printf ("Time for writing all to the Harddisk: %f s\n", write_time);
	printf("Done!\n");
}

template<class ALGORITHM>
Rainbow_Cuda_Cracker<ALGORITHM>::~Rainbow_Cuda_Cracker() { }

}
