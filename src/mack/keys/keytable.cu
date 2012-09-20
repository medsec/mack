#include <mack/keys/keytable.cuh>
#include <mack/core/text_file.hpp>

namespace mack{

// Keytable constructor
// chartable      : char[] containing all valid characters
// number_of_chars: length of chartable
__host__ __device__
Keytable::Keytable(const unsigned char* char_table, unsigned int number_of_chars)
	: char_table_(char_table),
		num_chars_(number_of_chars) {}

// Copy Constructor for keytable
// keytable: the keytable to copy
__host__ __device__
Keytable::Keytable(Keytable const& keytable)
	: char_table_(keytable.char_table_),
		num_chars_(keytable.num_chars_) {}

// Getter
// return: get the number of characters in this table
__host__ __device__
unsigned int const&
Keytable::get_number_of_characters() const
{
	return num_chars_;
}

// Getter
// return: get the character of given index in this table
__host__ __device__
unsigned char const&
Keytable::get_character(unsigned char index) const
{
	return char_table_[index];
}

// Index Getter
// return: the index of the char
__host__ __device__
unsigned int
Keytable::get_character_index(const unsigned char character) const
{
	return ((character - char_table_[0]) % num_chars_);
}


// Read a character table from a file
// filename       : the file to read from
// number_of_chars: the number of characters that are read
// return         : the read character table
unsigned char*
char_table_read(char const* filename, unsigned int& number_of_chars)
{
	unsigned int number_of_lines = 0;
	char** lines = mack::core::text_file_read_lines(filename, number_of_lines, '#');
	number_of_chars = number_of_lines;
	if (lines == NULL) { return NULL; }
	unsigned char* char_table = (unsigned char*) malloc(number_of_chars * sizeof(unsigned char));
	unsigned int i = 0;
	while (i < number_of_chars) {
		char_table[i] = atoi(lines[i]);
		++i;
	}
	mack::core::text_file_delete_lines(lines, number_of_chars);
	return char_table;
}

}
