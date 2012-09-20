#ifndef __MACK_LOGGING_STREAM_LOGGER_HPP__
#define __MACK_LOGGING_STREAM_LOGGER_HPP__

#include <string>
#include <iostream>

#include <mack/logging/logger.hpp>
#include <mack/options/values.hpp>

namespace mack {
namespace logging {

/**
 * @brief A logger for printing log messages to output streams.
 * @is_of_type{loggers}
 * @option{o,output,std::cerr}
 * The output stream to be used. Either std::cout or std::cerr.
 *
 * @author Johannes Kiesel
 * @date Aug 10 2012
 * @see mack::logging::logger
 */
class stream_logger : public logger {

  public:

    /**
     * @brief The long flag of the output option.
     */
    static const std::string flag_output;

    /**
     * @brief The keyword for selecting standard out as output.
     */
    static const std::string selection_output_stdout;

    /**
     * @brief The keyword for selecting standard error as output.
     */
    static const std::string selection_output_stderr;

    /**
     * @brief Constructor for mack/options.
     * @param values The values containing the output option
     */
    stream_logger(
        mack::options::values const* values);

    /**
     * @brief The constructor.
     * @param output_stream The stream to append log messages to
     */
    stream_logger(
        std::ostream& output_stream);

    virtual
    ~stream_logger();

    virtual
    void
    log(
        std::string const& message);

  private:

    /**
     * @brief The default constructor.
     */
    stream_logger();

    /**
     * @brief The output stream to append log messages to.
     */
    std::ostream* _output;

}; // class stream_logger

} // namespace logging
} // namespace mack

#endif /* __MACK_LOGGING_STREAM_LOGGER_HPP__ */

