/**
 * @file files.hpp
 * @brief Methods for reading and writing files.
 * @author Johannes Kiesel
 * @date Jule 9th, 2012
 */
#ifndef _MACK_CORE_FILES_HPP_
#define _MACK_CORE_FILES_HPP_

#include <string>
#include <exception>
#include <boost/exception/exception.hpp>
#include <boost/exception/error_info.hpp>
#include <boost/exception/info.hpp>
#include <boost/filesystem.hpp>

namespace mack {
namespace core {

/**
 * @namespace mack::core::files
 * @brief A namespace containing methods for reading from and writing to files.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 */
namespace files {

/**
 * @brief An exception which is thrown if an error occurred while processing
 * a file.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 * @see errinfo_file
 */
struct file_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An exception indicating that a required file does not exist.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 * @see errinfo_file the file which does not exist
 */
struct file_not_exists_error : virtual file_error {
};

/**
 * @brief An exception indicating that an object that was considered to be a
 * file is none.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 * @see errinfo_file the object that is not a file
 */
struct not_a_file_error : virtual file_error {
};

/**
 * @brief An exception indicating that an object that was considered to be a
 * directory is none.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 * @see errinfo_file the object which is not a directory
 */
struct not_a_directory_error : virtual file_error {
};

/**
 * @brief An exception indicating that an error occured while reading a file.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 * @see errinfo_file the file which was read
 */
struct file_read_error : virtual file_error {
};

/**
 * @brief An exception indicating that an error occured while writing to a file.
 * @author Johannes Kiesel
 * @date June 24th, 2012
 * @see errinfo_file the file which was written
 */
struct file_write_error : virtual file_error {
};

/**
 * @brief An exception which is thrown if an error occurred while parsing a file.
 * @author Johannes Kiesel
 * @date Jule 12th, 2012
 * @see errinfo_file the file which was parsed
 * @see errinfo_parse_cause a description of what was wrong
 * @see errinfo_parse_line the line of the error in the file
 */
struct parse_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error_info for indicating the cause of throwing a parse_error.
 *
 * @author Johannes Kiesel
 * @date Jule 12th, 2012
 */
typedef boost::error_info<struct tag_parse_error_cause, std::string>
  errinfo_parse_cause;

/**
 * @brief An error_info for indicating the file line in which parsing failed.
 *
 * @author Johannes Kiesel
 * @date Jule 12th, 2012
 */
typedef boost::error_info<struct tag_parse_error_line, size_t>
  errinfo_parse_line;

/**
 * @brief An error_info for indicating a file.
 *
 * @author Johannes Kiesel
 * @date Jule 12th, 2012
 */
typedef boost::error_info<struct tag_file, std::string>
  errinfo_file;

/**
 * @brief Reads the content of a file into a string.
 * @param file_path the path of the file to be read
 * @return the content of the file
 * @throws file_not_exists_error if no file with given file_path exists
 * @throws not_a_file_error if the object at given file_path is not a regular
 * file
 * @throws file_read_error if an error occured on reading the file
 */
std::string
read_file(
    boost::filesystem::path const& file_path);

/**
 * @brief Writes a text to a file.
 * @param content the text to be written
 * @param file_path the path of the output file
 * @throws file_not_exists_error if the parent of given file_path
 * does not exist
 * @throws not_a_directory_error if the parent of given file_path
 * is not a directory
 * @throws file_write_error if an error occured on writing the file
 */
void
write_file(
    std::string const& content,
    boost::filesystem::path const& file_path);

/**
 * @brief Checks if a text matches the content of a file.
 * @param file_path the file to check the content of
 * @param content the text to compare the content of the file with
 * @return <tt>true</tt> if the content of the file is equal to given text
 * @throws file_not_exists_error if no file with given file_path exists
 * @throws not_a_file_error if the object at given file_path is not a regular
 * file
 * @throws file_read_error if an error occured on reading the file
 */

bool
is_content_of(
    boost::filesystem::path const& file_path,
    std::string const& content);

/**
 * @brief Gets an error message for a file_not_exists_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    file_not_exists_error const& error);

/**
 * @brief Gets an error message for a not_a_directory_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    not_a_directory_error const& error);

/**
 * @brief Gets an error message for a not_a_file_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    not_a_file_error const& error);

/**
 * @brief Gets an error message for a file_read_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    file_read_error const& error);

/**
 * @brief Gets an error message for a file_write_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    file_write_error const& error);

/**
 * @brief Gets an error message for a parse_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    parse_error const& error);

} // namespace files
} // namespace core
} // namespace mack

#endif // _MACK_CORE_FILES_HPP_
