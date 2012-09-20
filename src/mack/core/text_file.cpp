// Reading all lines of text files
// author: johannes
//
#include <mack/core/text_file.hpp>

namespace mack{
namespace core{

// Reading all lines
// filename       : file to read from
// number_of_lines: number of lines read
// line_comment   : not reading lines starting with this char
// return         : array of lines
char**
text_file_read_lines(
		const char* filename,
		unsigned int& number_of_lines,
		char line_comment)
{
	number_of_lines = 0;
	char** lines = NULL;
	FILE* file = fopen((const char*)filename, "r");
	if (file == NULL) {
		return NULL;
	} else {
		// first run: get number_of_lines
		while (1) {
			int c = fgetc(file);
			if (c != line_comment) {
				++number_of_lines;
			}
			while (c != '\n' && c != EOF) {
				c = fgetc(file);
			}
			if (c == EOF) { break; }
		}
		--number_of_lines;
		rewind(file);

		lines = (char**) malloc(number_of_lines * sizeof(char*));
		unsigned int line = 0;
		// second run: get lines
		while (line < number_of_lines) {
			int c = fgetc(file);
			if (c == line_comment) {
				while (c != '\n' && c != EOF) {
					c = fgetc(file);
				}
			} else {
				long int line_length = 0;
				while (c != '\n' && c != EOF) {
					c = fgetc(file);
					++line_length;
				}
				fseek(file, -(line_length + 1), SEEK_CUR);
				lines[line] = (char*) malloc((line_length + 1) * sizeof(char));
				long int i = 0;
				while (i < line_length) {
					lines[line][i] = (char) fgetc(file);
					++i;
				}
				lines[line][i] = 0;
				++line;
				c = fgetc(file);
			}
			if (c == EOF) { break; }
		}

		// clean up
		fclose(file);

		return lines;
	}
}


// Reading all lines
// filename       : file to read from
// number_of_lines: number of lines read
// return         : array of lines
char**
text_file_read_lines(const char* filename, unsigned int& number_of_lines)
{
	number_of_lines = 0;
	char** lines = NULL;
	FILE* file = fopen((const char*)filename, "r");
	if (file == NULL) {
		return NULL;
	} else {
		// first run: get number_of_lines
		while (1) {
			int c = fgetc(file);
			++number_of_lines;
			while (c != '\n' && c != EOF) {
				c = fgetc(file);
			}
			if (c == EOF) { break; }
		}
		--number_of_lines;
		rewind(file);

		lines = (char**) malloc(number_of_lines * sizeof(char*));
		unsigned int line = 0;
		// second run: get lines
		while (line < number_of_lines) {
			int c = fgetc(file);
			long int line_length = 0;
			while (c != '\n' && c != EOF) {
				c = fgetc(file);
				++line_length;
			}
			fseek(file, -(line_length + 1), SEEK_CUR);
			lines[line] = (char*) malloc((line_length + 1) * sizeof(char));
			long int i = 0;
			while (i < line_length) {
				lines[line][i] = (char) fgetc(file);
				++i;
			}
			lines[line][i] = 0;
			++line;
			c = fgetc(file);
			if (c == EOF) { break; }
		}

		// clean up
		fclose(file);

		return lines;
	}
}

// Print all lines to stdout
// lines          : the lines to be printed
// number_of_lines: length of lines
void
text_file_print(char** lines, unsigned int number_of_lines)
{
	unsigned int i = 0;
	while (i < number_of_lines) {
		printf("%s\n", lines[i]);
		++i;
	}
}

// Free memory
// lines          : the lines
// number_of_lines: length of lines
void
text_file_delete_lines(char** lines, unsigned int number_of_lines)
{
	unsigned int i = 0;
	while (i < number_of_lines) {
		free(lines[i]);
		++i;
	}
	free(lines);
}

}
}
