#ifndef __MACK_OPTIONS_CONFIGURATION_FILE_HPP__
#define __MACK_OPTIONS_CONFIGURATION_FILE_HPP__

#include <mack/options/program_options.hpp>

#include <boost/filesystem/path.hpp>

#include <vector>
#include <string>

namespace mack {
namespace options {

/**
 * @namespace mack::options::configuration_file
 * @brief Methods for parsing configuration files for program options.
 * @details Configuration files have the following syntax rules:
 * <ul>
 * <li>Each not empty line is seen as a command.</li>
 * <li>Commands can be
 *  <ul>
 *  <li><i>\<option flag\></i></li>
 *  <li><i>\<option flag\></i> = <i>\<value\></i></li>
 *  </ul>
 *  for assigning a *value* (or the default value in the first case) to the
 *  option with the specified *option flag*.
 * </li>
 * <li>Comments start with a '#' and extend to the end of the line.</li>
 * <li>Leading and trailing whitespaces of *option flag* and *value* are
 * omitted.</li>
 * </ul>
 *
 * @author Johannes Kiesel
 * @date Aug 26 2012
 */
namespace configuration_file {

/**
 * @brief Parses and interprets a configuration file.
 * @param file_path the path of the configuration file
 * @param options the program options to be configured
 * @throws ::mack::core::null_pointer_error if options is <tt>null</tt>
 * @throws invalid_flag_error if a parsed flag is not a valid flag. A valid
 * flag consists of letters and digits at any position and dashes and
 * underscored at any but the first position
 * @throws no_such_namespace_error if a parsed option namespace does not exist
 * @throws no_such_option_error if an option with the specified flag does
 * not exist in the option namespace
 * @throws option_collision_error if no namespace is specified, no option
 * with the specified flag exists in the default namespace, but multiple
 * options with the specified flag exist in other namespaces
 * @throws invalid_value_error if a parsed value is not a valid one
 * for the specified option
 * @throws ::mack::core::files::file_not_exists_error if the configuration file
 * option is referenced, but the path specified by the value does not exist
 * @throws ::mack::core::files::not_a_file_error if the configuration file
 * option is referenced, but the path specified by the value is not a file
 * @throws ::mack::core::files::file_read_error if the configuration file
 * option is referenced, but an error occured while reading the file which
 * is specified by the value
 * @throws no_value_error if the the configuration file
 * option is referenced, which requires a value at this step,
 * but no value is assigned to it
 * @see mack::options::configuration_file
 */
void
parse(
    boost::filesystem::path const& file_path,
    program_options* options);

/**
 * @brief Parses and interprets the content of a configuration file.
 * @param content the content of the configuration file
 * @param options the program options to be configured
 * @throws ::mack::core::null_pointer_error if options is <tt>null</tt>
 * @throws invalid_flag_error if a parsed flag is not a valid flag. A valid
 * flag consists of letters and digits at any position and dashes and
 * underscored at any but the first position
 * @throws no_such_namespace_error if a parsed option namespace does not exist
 * @throws no_such_option_error if an option with the specified flag does
 * not exist in the option namespace
 * @throws option_collision_error if no namespace is specified, no option
 * with the specified flag exists in the default namespace, but multiple
 * options with the specified flag exist in other namespaces
 * @throws invalid_value_error if a parsed value is not a valid one
 * for the specified option
 * @throws ::mack::core::files::file_not_exists_error if the configuration file
 * option is referenced, but the path specified by the value does not exist
 * @throws ::mack::core::files::not_a_file_error if the configuration file
 * option is referenced, but the path specified by the value is not a file
 * @throws ::mack::core::files::file_read_error if the configuration file
 * option is referenced, but an error occured while reading the file which
 * is specified by the value
 * @throws no_value_error if the the configuration file
 * option is referenced, which requires a value at this step,
 * but no value is assigned to it
 * @see mack::options::configuration_file
 */
void
parse(
    std::string const& content,
    program_options* options);

/**
 * @brief Parses and interprets one line of a configuration file.
 * @param line the content of the line
 * @param options the program options to be configured
 * @throws ::mack::core::null_pointer_error if options is <tt>null</tt>
 * @throws invalid_flag_error if a parsed flag is not a valid flag. A valid
 * flag consists of letters and digits at any position and dashes and
 * underscored at any but the first position
 * @throws no_such_namespace_error if a parsed option namespace does not exist
 * @throws no_such_option_error if an option with the specified flag does
 * not exist in the option namespace
 * @throws option_collision_error if no namespace is specified, no option
 * with the specified flag exists in the default namespace, but multiple
 * options with the specified flag exist in other namespaces
 * @throws invalid_value_error if a parsed value is not a valid one
 * for the specified option
 * @throws ::mack::core::files::file_not_exists_error if the configuration file
 * option is referenced, but the path specified by the value does not exist
 * @throws ::mack::core::files::not_a_file_error if the configuration file
 * option is referenced, but the path specified by the value is not a file
 * @throws ::mack::core::files::file_read_error if the configuration file
 * option is referenced, but an error occured while reading the file which
 * is specified by the value
 * @throws no_value_error if the the configuration file
 * option is referenced, which requires a value at this step,
 * but no value is assigned to it
 * @see mack::options::configuration_file
 */
void
parse_line(
    std::string const& line,
    program_options* options);

} // namespace configuration_file
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_CONFIGURATION_FILE_HPP__ */
