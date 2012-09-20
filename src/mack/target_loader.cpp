#include <mack/target_loader.hpp>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>

namespace mack{
namespace targetloaders{
	mack::targetloaders::Target_Loader::~Target_Loader(){

	}

	/**
	 * Initializes the current target loader with correct target size.
	 * Argument: target_size - the size of the hashes to search for
	 */
	void
	mack::targetloaders::Target_Loader::init(size_t target_size)
	{
		_target_count = 0;
		_target_size = target_size;
	}

	/**
	 * Counts all lines in a file. This means that even lines with only a '\\n' also counted.
	 * Argument: fil_path, the path to the file
	 */

	size_t
	mack::targetloaders::Target_Loader::line_count(const char* file_path)
	{
		std::ifstream file(file_path);

	    if(!file){
	    	printf("Warning: target file %s does not exist!\n",file_path);
	    	perror("Terminating");
	    	exit(EXIT_FAILURE);
	    }

		file.seekg (0, std::ios::beg);
		int i = 0;
		std::string line;
		while(file.good()){
			getline(file,line);
			++i;
		}
		file.clear();
		file.seekg (0, std::ios::beg);

		return i;
	}

	/**
	 * Returns the number of the targets.
	 */
	size_t
	mack::targetloaders::Target_Loader::
	get_target_count(){
		return _target_count;
	}

	/**
	 * Returns the i-th target with target_size from targets. If the index is out of bounds, mack will be terminated.
	 */
	unsigned char*
	mack::targetloaders::Target_Loader::
	get_target(size_t index, unsigned char* targets, size_t target_size){
		if(index >= _target_count){
			printf("Target index %lu is out of bounds. Number of loaded targets: %lu\n", index, _target_count);
			exit(EXIT_FAILURE);
		}
		return &targets[0 + index*target_size];
	}

	/**
	 * Return the i-th target with known target size. If the index is out of bounds, mack will be terminated.
	 */
	unsigned char*
	mack::targetloaders::Target_Loader::
	get_target(size_t index, unsigned char* targets){
		if(index >= _target_count){
			printf("Target index %lu is out of bounds. Number of loaded targets: %lu\n", index, _target_count);
			exit(EXIT_FAILURE);
		}
		return get_target(index, targets, _target_size);
	}

}
}

