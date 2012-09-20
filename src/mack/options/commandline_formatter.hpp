#ifndef __MACK_OPTIONS_COMMANDLINE_FORMATTER_HPP__
#define __MACK_OPTIONS_COMMANDLINE_FORMATTER_HPP__

#include <mack/options/option.hpp>
#include <mack/options/options.hpp>

#include <iostream>
#include <ostream>

#include <string>

namespace mack {
namespace options {

/**
 * @brief A text formatter for command line output.
 * @details Instances of this class can print information on option or options
 * objects to an output stream. The maximum line length and tab stop size can
 * be adjusted as well as some further settings.
 *
 * @author Johannes Kiesel
 * @date Aug 24 2012
 * @see mack::options
 */
class commandline_formatter {

  public:

    /**
     * @brief The constructor.
     * @param output_stream the stream to write the formatted data to
     * @param print_flag_prefix if flag prefixes (<tt>-\-</tt>) should be
     * printed in front of each option flag
     * @param print_full_descriptions if full descriptions should be printed
     * instead of only the brief descriptions
     * @param line_length_chars the maximum number of characters in a line
     * @param tabsize the number of space characters each level of indentation
     * consists of
     */
    commandline_formatter(
        std::ostream& output_stream = std::cout,
        const bool print_flag_prefix = true,
        const bool print_full_descriptions = true,
        const size_t line_length_chars = 80,
        const size_t tabsize = 2);

    /**
     * @brief The destructor.
     */
    ~commandline_formatter();

    /**
     * @brief Print information about an option to the output stream.
     * @param option the option
     * @param option_namespace the namespace of the option if it is not the
     * root namespace
     */
    void
    print(
        option const* option,
        std::string const& option_namespace = "");

    /**
     * @brief Print little information about an option to the output stream.
     * @param option the option
     * @param option_namespace the namespace of the option if it is not the
     * root namespace
     */
    void
    print_brief(
        option const* option,
        std::string const& option_namespace = "");

    /**
     * @brief Print information about an options object to the output stream.
     * @param options the options
     * @param option_namespace the namespace of the options if it is not the
     * root namespace
     */
    void
    print(
        options const* options,
        std::string const& option_namespace = "");

    /**
     * @brief Print little information about an options object to the output
     * stream.
     * @param options the options
     * @param option_namespace the namespace of the options if it is not the
     * root namespace
     */
    void
    print_brief(
        options const* options,
        std::string const& option_namespace = "");

    /**
     * @brief Print only the description of an option to the output stream.
     * @param option the option
     */
    void
    print_description(
        option const* option);

    /**
     * @brief Print only the description of a program to the output stream.
     * @details Either the brief or the full description will be printed along
     * with the program name.
     * @param program_name the name of the program
     * @param brief_description a brief description of the program
     * @param full_description a description of the program
     */
    void
    print_description(
        std::string const& program_name,
        std::string const& brief_description,
        std::string const& full_description);

    /**
     * @brief Print only the description of an options object to the output
     * stream.
     * @param options the options
     */
    void
    print_description(
        options const* options);

  private:

    void
    print_option_name(
        option const* option,
        std::string const& option_namespace,
        const bool with_value = false);

    void
    print_attributes(
        option const* option);

    void
    print_children(
        option const* option,
        std::string const& option_namespace);

    void
    increase_indent(
        const size_t amount = 1);

    void
    decrease_indent(
        const size_t amount = 1);

    void
    print(
        std::string const& text);

    void
    print_line(
        std::string const& text = "");

    std::ostream& _output_stream;

    bool _print_flag_prefixes;

    bool _print_full_descriptions;

    size_t _line_length_chars;

    size_t _tabsize;

    std::string _indent;

    size_t _already_filled;

}; // class commandline_formatter

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_COMMANDLINE_FORMATTER_HPP__ */

