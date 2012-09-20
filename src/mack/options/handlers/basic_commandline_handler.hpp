#ifndef __MACK_OPTIONS_HANDLERS_BASIC_COMMANDLINE_HANDLER_HPP__
#define __MACK_OPTIONS_HANDLERS_BASIC_COMMANDLINE_HANDLER_HPP__

#include <mack/options/handlers/handler.hpp>

#include <mack/options/values.hpp>

#include <string>
#include <iostream>

namespace mack {
namespace options {
namespace handlers {

/**
 * @brief A simple interaction handler which reads input from a stream.
 * @is_of_type{option_handlers}
 * @option_switch{r,repeat}
 * If this option is set, the interaction menu will be shown again after the
 * first run of the program has completed.
 *
 * @author Johannes Kiesel
 * @date Aug 11 2012
 * @see mack::options
 * @see mack::options::handlers::handler
 */
class basic_commandline_handler : public handler {

  public:

    /**
     * @brief The complete constructor.
     * @param repeat if the handler can be used more than once
     * @param input_stream the stream to read commands from
     * @param message_output_stream the stream to write normal messages to
     * @param error_output_stream the stream to write error messages to
     */
    basic_commandline_handler(
        const bool repeat = false,
        std::istream& input_stream = std::cin,
        std::ostream& message_output_stream = std::cout,
        std::ostream& error_output_stream = std::cerr);

    /**
     * @brief The constructor for use within programs.
     * @param values the values for configuring the handler
     */
    basic_commandline_handler(
        values const* values);

    virtual
    ~basic_commandline_handler();

    virtual
    values const*
    run(); // throws exit_requested

  private:

    values const*
    execute_command(
        std::string const& command) const;

    void
    set_value_from_input(
        std::string const& input);

    void
    print_options() const;

    bool _has_been_run;

    bool _repeat;

    std::istream& _input_stream;

    std::ostream& _output_stream;

    std::ostream& _error_stream;

}; // class basic_commandline_handler

} // namespace handlers
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_HANDLERS_BASIC_COMMANDLINE_HANDLER_HPP__ */

