/**
 * @file mack/keys/key.cuh
 */

#ifndef KEY_HPP_
#define KEY_HPP_

#include "keytable.cuh"
#include <stdlib.h>
#include <stdio.h>

namespace mack{

/**
 * @class Key
 * @brief This class represents a key.
 * 
 * @details It contains two arrays, one for the text in readable char table characters and
 * one for the indices.
 * Furthermore it provides methods for incrementing the key, getters and setters and
 * a length field.
 * 
 * @author Johannes Kiesel
 * @date 20.05.2012
 * @version 0.1
 */

class Key
{
	public:
		/** 
		 * @brief Constructor for a key
		 * @param key_table defines the space of valid keys
		 * @param key_length the number of characters the key is long
		*/
		__host__ __device__ Key(Keytable const* key_table, unsigned int key_length);

		/**
		 * @brief Copy Constructor for key
		 * @param key the key to copy the state from
		 */ 
		__host__ __device__ Key(Key const& key);

		/**
		 * @brief Destructor
		 */ 
		__host__ __device__ ~Key();

		/**
		 * @brief Getter to get the length of the key.
		 * @returns the number of characters of this key
		 */ 
		__host__ __device__ unsigned int const& get_length() const;


		/**
		 * @brief Getter, to get the key string in char table coding
		 * @returns current character sequence
		 */ 
		__host__ __device__ unsigned char const* get_key_string() const;

		/**
		 * @brief Setter, set a new keystring.
		 * @param new_key the new key string in char, it will be translated in char table coding
		 */
		__host__ __device__ void set_key_string(const unsigned char* new_key);

		/**
		 * @brief DEBUG - Getter, in future releases this function will be removed
		 */
		__host__ __device__ const unsigned char* get_indices() const;

		/**
		 * @brief Increment this key
		 * @returns 1 if the key was successfully incremented and
		 * @returns 0 if an overflow occurred
		 */ 
		__host__ __device__ int increment();

		/**
		 * @brief Get the next key (incremented by certain amount)
		 * @param amount the next key by this amount
		 * @returns 1 if the key was successfully incremented and
		 * @returns 0 if an overflow occurred;
		 */ 
		__host__ __device__ int increment(unsigned int amount);

	private:
		unsigned int length_;
		const Keytable* table_;
		unsigned char* chars_;
		unsigned char* indices_;
};


/**
 * @brief Gets a printable (and null-terminated) char array for the key
 * @param key the key to get the current sequence of
 * @returns the char sequence
 */ 
__host__ __device__ char* malloc_character_key_string(Key const& key);

/** 
 * @brief Prints the current sequence of the key (and only it) to stdout
 * @param key the key to get the current sequence of
 */
__host__ __device__ void print_character_key_string(Key const& key);

}

#endif /* KEY_HPP_ */
