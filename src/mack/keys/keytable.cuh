/*
 * keytable.hpp
 *
 *  Created on: 20.06.2012
 *      Author: paul
 */

#ifndef KEYTABLE_HPP_
#define KEYTABLE_HPP_

#include <stdlib.h>
#include <cuda_runtime.h>

namespace mack{

/**
 * @file /mack/keys/keytable.cuh
 * @class Keytable
 * @brief The Keytable class represents a character table.
 * @details This class helps to use multiple key tables with the same code. We have only to load an other chartable from file.
 * 
 * @date 20.06.2012
 * @author Johannes Kiesel
 * @authos Paul Kramer
 */

class Keytable
{
	public:
		/*
		 * @brief Keytable constructor, constructs a new Keytbale object holing a character table
		 * @param chartable char[] containing all valid characters
		 * @param number_of_chars length of chartable
		 */ 
		__host__ __device__ Keytable(const unsigned char* char_table, unsigned int number_of_chars);

		/** 
		 * @brief Copy Constructor for chartable
		 * @param keytable the keytable to copy
		 */
		__host__ __device__ Keytable(Keytable const& keytable);

		/**
		 * @brief Getter to get the number of characters in this chartable
		 * @returns get the number of characters in this table
		 */
		__host__ __device__ unsigned int const& get_number_of_characters() const;

		/**
		 * @brief Getter to get a special character from the table
		 * @returns get the character of given index in this table
		 */
		__host__ __device__ unsigned char const& get_character(unsigned char index) const;

		/**
		 * @brief Getter to get the index of a given character.
		 * @details Do not use this often, because it is very slow.
		 * @returns the index of the given character 
		 */ 
		__host__ __device__ unsigned int get_character_index(const unsigned char character) const;

	private:
		unsigned char const* char_table_;
		unsigned int num_chars_;

		Keytable() { }
};

/**
 * @brief Reads a character table from a file given by filename
 * @param filename the file to read from
 * @param number_of_chars the number of characters that are read
 * @returns the read character table
 */ 
unsigned char* char_table_read(char const* filename, unsigned int& number_of_chars);

}

#endif /* KEYTABLE_HPP_ */
