/*
 * container.cpp
 *
 *  Created on: 31.08.2012
 *      Author: azzaroff
 */

#include "container_callback.h"

namespace mack {
namespace callbacks {

container_callback::container_callback() {
	// TODO Auto-generated constructor stub

}

container_callback::~container_callback() {
	// TODO Auto-generated destructor stub
}

void
container_callback::call(unsigned char* target, mack::core::candidate* candidate, unsigned int targetlength)
{
	//temp map target which contains the hash in string hex representation
	char tmp[2*targetlength+1];
	tmp[2*targetlength] = 0;

	//printf("found '%s':", candidate->value);
	for(unsigned int i = 0 ; i < targetlength; ++i){
			sprintf(&tmp[2*i],"%02x", target[i] & 0xff); //02 fills leading zeros
		}
	//printf("%s \n", tmp);

	std::string map_target(tmp);
	_container[map_target] = (*candidate);
}

std::map<std::string,mack::core::candidate>
container_callback::get_container()
{
	return _container;
}

}
}

