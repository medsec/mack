#ifndef __MACK_OPTIONS_PARSER_HPP__
#define __MACK_OPTIONS_PARSER_HPP__

#include <string>

#include <iostream>

#include <mack/options/values.hpp>
#include <mack/options/program_options.hpp>
#include <mack/options/handlers/handler.hpp>

namespace mack {
namespace options {

/**
 * @brief The program option parser for *Mack the Knife*.
 * @details The parser is the interface between a program and the options
 * framework. It will take care of all necessary steps of parsing the program
 * options as they are specified in the documentation of the program.
 *
 * It is used as shown in this example for a program "my_program":
 * \verbinclude quickstart_program.hpp
 * 
 * Note that only the created objects within the values object have to be
 * deleted. The constructor and the <tt>parse</tt> method will throw an
 * \ref exit_requested in the case of a fatal error and if the user decided
 * to quit the program.
 *
 * @author Johannes Kiesel
 * @date Aug 28 2012
 * @see mack::options
 */
class parser {

  public:

    /**
     * @brief Creates a new program option parser.
     * @param argc the number of arguments in argv
     * @param argv the program arguments as array of c strings as they are
     * given to main; therefore, also with the program name as first
     * argument
     * @param program_name the name of the program as specified with the
     * \@program annotation
     * that occur during the creation of the parser
     * @param message_output_stream a stream to write a help message to, if one
     * is requested
     * @param error_output_stream a stream to write error messages of errors
     * that occur during the creation of the parser
     * @throws exit_requested if a fatal error occurred or the user decided to
     * quit
     */
    parser(
        const int argc,
        char const* const* argv,
        std::string const& program_name,
        std::ostream& message_output_stream = std::cout,
        std::ostream& error_output_stream = std::cerr);

    /**
     * @brief Creates a new program option parser with specified program
     * options.
     * @details This constructor is used for testing. Use the other
     * constructor instead.
     * @param argc the number of arguments in argv
     * @param argv the program arguments as array of c strings as they are
     * given to main; therefore, also with the program name as first
     * argument
     * @param program_options the options
     * @param message_output_stream a stream to write a help message to, if one
     * is requested
     * @param error_output_stream a stream to write error messages of errors
     * that occur during the creation of the parser
     * @throws ::mack::core::null_pointer_error if the program options are
     * <tt>null</tt>
     * @throws exit_requested if a fatal error occurred or the user decided to
     * quit
     */
    parser(
        const int argc,
        char const* const* argv,
        program_options* program_options,
        std::ostream& message_output_stream = std::cout,
        std::ostream& error_output_stream = std::cerr);

    /**
     * @brief The destructor.
     * @details Deletes the last parsed values, the program options and
     * the interaction handler.
     */
    ~parser();

    /**
     * @brief Parses a new set of values for the program options of the
     * program.
     * @details This will use a (via commandline arguments) specified
     * interaction handler to create a new setting for the options. Then it
     * will create the specified values and return them.
     *
     * If values have
     * already been created by a previous call to this method, these previously
     * created values will be deleted. Note however, that the objects within
     * the values are not deleted by this method, but need to be deleted in
     * the program code.
     * @return values the created values
     * @throws exit_requested if a fatal error occurred or the user decided to
     * quit (see \ref exit_requested.exit_code)
     */
    values const*
    parse();

  private:

    parser();

    void
    check_help();

    void
    set_program(
        std::string const& program_name);

    void
    set_program_options(
        const int argc,
        char const* const* argv);

    void
    set_interaction_handler();

    values const* _values;

    program_options* _program_options;

    mack::options::handlers::handler* _interaction_handler;

    std::ostream& _message_stream;

    std::ostream& _error_stream;

}; // class parser

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_PARSER_HPP__ */

