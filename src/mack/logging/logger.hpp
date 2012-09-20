#ifndef __MACK_LOGGING_LOGGER_HPP__
#define __MACK_LOGGING_LOGGER_HPP__

#include <string>

/**
 * @option_type{loggers,Loggers}
 * @brief Loggers for handling log messages.
 * @details There are four logging levels for Mack the Knife:
 *  * **debug** Frequent state information for tracking bugs.
 *  * **info** Meaningful state messages.
 *  * **warning** Information of unnormal behaviour which might cause undesired
 *  behaviour.
 *  * **error** Information of an occurred error.
 * @option_type_class{mack::logging::logger}
 */

namespace mack {
namespace logging {

/**
 * @brief Abstract base class for loggers.
 * @details A logger is a strategy for processing log messages to certain
 * outputs. Each logger within the *Mack the Knife* framework has to extend
 * this class and to contain the *\@is_of_type{loggers}* annotation in its main
 * documentation block.
 *
 * Logging is performed by employing the pure virtual <tt>log</tt> method of
 * this class. For more information on how to use the logging system see the
 * documentation of \ref mack::logging.
 *
 * @author Johannes Kiesel
 * @date Aug 10 2012
 *
 * @see log(std::string const& message)
 * @see mack::logging
 */
class logger {

  public:

    /**
     * @brief The default constructor.
     */
    logger();

    /**
     * @brief The destructor.
     */
    virtual
    ~logger();

    /**
     * @brief Outputs the logging message.
     * @details The way the message is logged depends on the concrete
     * implementation of the logger.
     *
     * No exceptions should be thrown by this message. Instead, the name of the
     * exception as well as the message should be displayed to standard error.
     */
    virtual
    void
    log(
        std::string const& message) = 0;

}; // class logger

} // namespace logging
} // namespace mack

#endif /* __MACK_LOGGING_LOGGER_HPP__ */

