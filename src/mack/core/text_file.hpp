/*
 * text_file.hpp
 *
 *  Created on: 20.06.2012
 *      Author: paul
 */

#ifndef TEXT_FILE_HPP_
#define TEXT_FILE_HPP_

#include <stdio.h>
#include <stdlib.h>

namespace mack{
namespace core{


// Reading all lines
// filename       : file to read from
// number_of_lines: number of lines read
// line_comment   : not reading lines starting with this char
// return         : array of lines
char** text_file_read_lines(
		const char* filename,
		unsigned int& number_of_lines,
		char line_comment);


// Reading all lines
// filename       : file to read from
// number_of_lines: number of lines read
// return         : array of lines
char** text_file_read_lines(const char* filename, unsigned int& number_of_lines);

// Print all lines to stdout
// lines          : the lines to be printed
// number_of_lines: length of lines
void text_file_print(char** lines, unsigned int number_of_lines);

// Free memory
// lines          : the lines
// number_of_lines: length of lines
void text_file_delete_lines(char** lines, unsigned int number_of_lines);

}
}

#endif /* TEXT_FILE_HPP_ */
