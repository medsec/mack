// Character Keys
// author: johannes
//
#include <mack/keys/key.cuh>

namespace mack{

/** Constructor for a key
* key_table : which defines the space of valid keys
* key_length: the number of characters the key is long
*/
__host__ __device__
mack::Key::Key(Keytable const* key_table, unsigned int key_length)
		:	length_(key_length), table_(key_table)
{
	chars_ = (unsigned char*) malloc(length_ * sizeof(unsigned char));
	indices_ = (unsigned char*) malloc(length_ * sizeof(unsigned char));
	unsigned char first_char = key_table->get_character(0);
	unsigned int index = 0;
	while (index < key_length) {
		chars_[index] = first_char;
		indices_[index] = 0;
		++index;
	}
}

/* Copy Constructor for key
* key: the key to copy the state from
*/
__host__ __device__
mack::Key::Key(Key const& key)
		: length_(key.length_), table_(key.table_) {
	chars_ = (unsigned char*) malloc(length_ * sizeof(unsigned char));
	indices_ = (unsigned char*) malloc(length_ * sizeof(unsigned char));
	unsigned int index = 0;
	while (index < length_) {
		indices_[index] = key.indices_[index];
		chars_[index] = key.chars_[index];
		++index;
	}
}

/*
 * Destructor
 */
__host__ __device__
mack::Key::~Key()
{
	free(chars_);
	free(indices_);
}

/*
 * Getter
 * return: the number of characters of this key
 */
__host__ __device__
unsigned int const&
mack::Key::get_length() const
{
	return length_;
}


/**
 * Getter
 * return: current character sequence
 */
__host__ __device__
const unsigned char*
mack::Key::get_key_string() const
{
	return chars_;
}

/**
 * Setter
 */
__host__ __device__
void
mack::Key::set_key_string(const unsigned char* new_key)
{
	for(unsigned int index = 0; index < length_; ++index) {
		indices_[index] = table_->get_character_index(new_key[index]);
		chars_[index] = table_->get_character(indices_[index]);
	}
}

/**
 * DEBUG Getter
 */
__host__ __device__
const unsigned char*
mack::Key::get_indices() const
{
	return indices_;
}

/**
 * Increment this key
 * return: 1 if the key was successfully incremented and
 *         0 if an overflow occurred
 */
__host__ __device__
int
mack::Key::increment()
{
	unsigned int index = 0;
	const unsigned int num_chars = table_->get_number_of_characters();
	while (index < length_) {
		indices_[index] = (indices_[index] + 1) % num_chars;
		chars_[index] = table_->get_character(indices_[index]);
		if (indices_[index] != 0) { break; }
		++index;
	}
	chars_[length_] = 0;
	indices_[length_] = 0;
	return index < length_;
}

/** Get the next key (incremented by certain amount)
 * amount: the next key by this amount
 * return: 1 if the key was successfully incremented and
 *         0 if an overflow occurred;
 */
__host__ __device__
int
mack::Key::increment(unsigned int amount)
{
	unsigned int index = 0;
	const unsigned int num_chars = table_->get_number_of_characters();
	while (index < length_) {
		const unsigned int byte_value = indices_[index] + amount;
		indices_[index] = byte_value % num_chars;
		chars_[index] = table_->get_character(indices_[index]);
		if (byte_value < num_chars) { return 1; }
		amount = byte_value / num_chars;
		++index;
	}
	return amount == 0;
}


/**
 * Gets a printable (and null-terminated) char array for the key
 * key   : the key to get the current sequence of
 * return: the char sequence
 */
__host__ __device__
char*
malloc_character_key_string(Key const& key)
{
	const unsigned int num_chars = key.get_length();
	char* character_string = (char*) malloc((num_chars + 1) * sizeof(char));
	unsigned char const* key_string = key.get_key_string();
	unsigned int index = 0;
	while (index < num_chars) {
		character_string[index] = (char) key_string[index];
		++index;
	}
	character_string[index] = 0;
	return character_string;
}

/**
 * Prints the current sequence of the key (and only it) to stdout
 * key   : the key to get the current sequence of
 */
__host__ __device__
void
print_character_key_string(Key const& key)
{
	char* character_string = malloc_character_key_string(key);
	printf("%s", character_string);
	free(character_string);
}

}
