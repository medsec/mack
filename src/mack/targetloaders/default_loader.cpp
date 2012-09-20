/*
 * default_loader.cpp
 *
 *  Created on: 06.09.2012
 *  Author: Felix Trojan
 *  Author: Paul Kramer
 */

#include <targetloaders/default_loader.hpp>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>

namespace mack{
namespace targetloaders{
	/**
	 * Empty destructor.
	 */
	mack::targetloaders::Default_Loader::~Default_Loader(){

	}

	/**
	 * Loads the file with given filename and returns a pointer to the targets in 32-bit representation. For some hash algorthims this can be
	 * helpful to increase the speed. Aditionally this method counts the number of valid targets and stores it.
	 */
	unsigned char*
	mack::targetloaders::Default_Loader::
	load_file_32(const char* file_path){
		typedef unsigned int uint32;

		//each line gets converted to a uint32 array
		//which is then converted to a unsigned char* array
		std::ifstream file(file_path);
		unsigned int number_of_lines = line_count(file_path);

		uint32* target = new uint32[_target_size/sizeof(uint32)];
		unsigned char* cbuffer = new unsigned char[number_of_lines*_target_size/sizeof(unsigned char)];
		memset(target,0, _target_size/sizeof(uint32));
		memset(cbuffer,0, number_of_lines*_target_size/sizeof(unsigned char));

		//maybe not all lines contain a target, so count the targets
		_target_count = 0;

		int i = 0;
	    std::string line;
		while(file.good()){
			getline(file,line);
			char* s = (char*)line.c_str();
			if(line.length() == _target_size*2)
			{
				//generates format strings for sscanf
				//sscanf coverts the char represantation of the hash to binary and stores them in buffer
				sscanf(s,"%8x",&target[0]);
				char scanf_str[8];
				for(int j = 1; j < (_target_size/sizeof(uint32)); ++j)
				{
					//creates the format string for sscanf depending on the hash width of the algorithm
					sprintf( scanf_str,"%%*%lux%%%lux", sizeof(uint32)*2*j, sizeof(uint32)*2 ); //8 because 8 hexchars = 2*4byte = 2*uint32
					sscanf(s,scanf_str,&target[j]);

				}
				for(int j = 0; j < (_target_size/sizeof(uint32)); ++j)
				{
					cbuffer[3 + j*sizeof(uint32) + i*_target_size] = (unsigned char)(target[j] & 0x000000ff);
					cbuffer[2 + j*sizeof(uint32) + i*_target_size] = (unsigned char)((target[j] >> 8) & 0x000000ff);
					cbuffer[1 + j*sizeof(uint32) + i*_target_size] = (unsigned char)((target[j] >> 16) & 0x000000ff);
					cbuffer[0 + j*sizeof(uint32) + i*_target_size] = (unsigned char)((target[j] >> 24) & 0x000000ff);
				}
				//count this target
				++_target_count;
			}
			++i;
		}
	    file.close();
	    delete[] target;
	    printf("%lu Targets loaded\n",_target_count);

	    return cbuffer;
	}

	/**
	 * Loads the file with given filename and returns a pointer to the targets in 8-bit representation. For some hash algorthims this can be
	 * helpful to increase the speed. Aditionally this method counts the number of valid targets and stores it.
	 */
	unsigned char*
	mack::targetloaders::Default_Loader::
	load_file_8(const char* file_path){
		//each line gets converted to a unsigned char array
		std::ifstream file(file_path);
		unsigned int number_of_lines = line_count(file_path);

		unsigned char* target = (unsigned char*) malloc(_target_size);
		unsigned char* cbuffer = (unsigned char*) malloc(number_of_lines*_target_size * sizeof(unsigned char));
		memset(target,0, _target_size);
		memset(cbuffer,0, number_of_lines*_target_size * sizeof(unsigned char));

		unsigned int i = 0;
	    std::string line;

	    //maybe not all lines contain a target, so count the targets
	    _target_count = 0;

		while(file.good()){
			getline(file,line);
			unsigned char* s = (unsigned char*)line.c_str();
			if(line.length() == _target_size*2)
			{
				for(int j = 0; j < (int)_target_size; ++j){
					sscanf((char*)(s+(2*j)), "%02x", (unsigned int*)(target+j));
				}
				memcpy((cbuffer+(i * _target_size)), target, _target_size);
				//count this line
				++_target_count;
			}
			++i;
		}
	    file.close();
	    if(target){
	    	free(target);
	    	target = 0;
	    }
	    printf("%lu Targets loaded\n",_target_count);
	    return cbuffer;
	}
}
}

