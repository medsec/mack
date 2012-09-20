/*
 * consoleoutput.cpp
 *
 *  Created on: 09.08.2012
 *      Author: paul
 */

#include "consoleoutput.h"

namespace mack {
namespace callbacks {

Console_Output::Console_Output() {
	// TODO Auto-generated constructor stub
}

Console_Output::~Console_Output() {
	// TODO Auto-generated destructor stub
}

void
Console_Output::call(unsigned char* target, mack::core::candidate* candidate, unsigned int targetlength)
{
	printf("found '%s':", candidate->value);
	for(unsigned int i = 0 ; i < targetlength; ++i){
			printf("%02x", target[i] & 0xff); //02 fills leading zeros
		}
	printf("\n");
}

}
}

