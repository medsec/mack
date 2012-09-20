
/*
 * rainbow_cuda_cracker_helper.cu
 *
 *  Created on: 26.06.2012
 *      Author: paul
 */

#include "rainbow_cuda_cracker_helper.cuh"

namespace mack{

/**
 * Reduction function for the rainbow cracker. The functions arguments:
 * 	- Key* key: the key to reduce
 * 	- int wordlength: the length of the key
 * 	- int factor1: the row or something like that
 * 	The reduction funktion add some integer value to the first byte of the key. Because of that
 * 	it is very possible that nearly every other bit of the key is flipped.
 */
__host__ __device__
void
reduce(Key* key, int wordlength, int factor1)
{
	key->increment(3*factor1 + ((wordlength-1) * 255));
}

/**
 * cudaMemCmp is a cuda memory compare function. It gets three arguments:
 * 	- unsigned char* left: the first word to compare with
 * 	- unsigned char* right: the other word
 * 	- int length: the maximum bytecount to compare
 * 	Result: int the function result is 0, if it does not match and 1 if it matches.
 */
__device__
int
cudaMemCmp(const unsigned char* left, const unsigned char* right, int length)
{
	int result = 1;
	while(result && (length > 0))
	{
		--length;
		result &= (left[length] == right[length]);
	}
	return result;
}

/**
 * string_to_unsigned_int changes the data type of given input string to unsigned int and return it.
 */
unsigned int string_to_unsigned_int(std::string input){
	std::stringstream ss;
	ss << input;
	unsigned int i;
	ss >> i;
	return i;
}

/**
 * This is a small helper function. Arguments:
 * 	- unsigned char* text
 * 	- int length
 * 	It outputs every byte of "text" as hex beginning with a 0x.
 */
__host__ __device__
void
printAsHex(unsigned char* text, int length)
{
	printf("0x");
	for(int i = 0 ; i < length; ++i){
		printf("%02x", text[i] & 0xff); //02 fills leading zeros
	}
}

/**
 * This function calculates the number of lines you need for standard rainbowtables, depending on the chainlength. Arguments:
 * 	- Keytable* keytable: the current keytable to get the number of chars
 * 	- unsigned int& chainlength: the length of the chain, maybe on can modify it in this method
 * 	- int& tablelength: the number of lines in the table
 * 	- int keylength: the length of the key
 *	This function calculates: 2^(2/3 * n), where n is ld(#chars in keytable) * length of key, as Hellman says.
 */
void get_table_dimensions(Keytable* keytable, unsigned int& chainlength, long long int& tablelength, int keylength)
{
	//(2 ^ log2(number_of_chars) * (2/3) * keylength)/chainlength
	tablelength = ceil(pow(2, (log(keytable->get_number_of_characters()) / log(2)) * (2.0f / 3.0f) * keylength) / chainlength);
}

/**
 * This method tests if the given file exists then it returns true, otherwise false.
 */
bool file_exists(const char * filename)
{
    if (FILE * file = fopen(filename, "r"))
    {
        fclose(file);
        return true;
    }
    return false;
}

}
