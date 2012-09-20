#ifndef __MACK_LOGGING_PREVIOUS_LOGGER_HPP__
#define __MACK_LOGGING_PREVIOUS_LOGGER_HPP__

#include <string>

#include <mack/logging/logger.hpp>

namespace mack {
namespace logging {

/**
 * @brief Not really a logger, but a type class to signalize that the
 * logger of the next lower log level should be used.
 * @is_of_type{loggers}
 *
 * @author Johannes Kiesel
 * @date Aug 10 2012
 * @see mack::logging::logger
 */
class previous_logger : public logger {

  public:

    previous_logger();

    virtual
    ~previous_logger();

    virtual
    void
    log(
        std::string const& message);

}; // class previous_logger

} // namespace logging
} // namespace mack

#endif /* __MACK_LOGGING_PREVIOUS_LOGGER_HPP__ */

