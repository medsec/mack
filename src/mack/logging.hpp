#ifndef __MACK_CORE_LOGGER_HPP__
#define __MACK_CORE_LOGGER_HPP__

/**
 * @file mack/logging.hpp
 */

#include <string>
#include <sstream>

#include <mack/logging/logger.hpp>

namespace mack {

/**
 * @namespace mack::logging
 * @brief Classes and methods for message logging.
 * @details Each program within the *Mack the Knife* framework automatically
 * supports this logging system. The system can be adjusted by default program
 * parameters.
 *
 * In order to use the system, the file <tt>mack/logging.hpp</tt> has to be
 * included. This enables the use of the logging macros, which support four
 * logging levels for indicating the type of the message: **debug**, **info**,
 * **warning** and **error**. Additionally, it is possibly to perform logging
 * by using a stringstream-like interface.
 *
 * @author Johannes Kiesel
 * @date Aug 22 2012
 * @see LOG_DEBUG(msg)
 * @see LOG_INFO(msg)
 * @see LOG_WARNING(msg)
 * @see LOG_ERROR(msg)
 * @see LOG_STREAM_DEBUG(msg)
 * @see LOG_STREAM_INFO(msg)
 * @see LOG_STREAM_WARNING(msg)
 * @see LOG_STREAM_ERROR(msg)
 */
namespace logging {

/**
 * @brief Sets the logger for all messages.
 * @details This logger will then be used for logging all messages.
 *
 * All logger objects which are no longer used for logging are deleted.
 * @param logger A pointer to the logger object to be used
 */
void
set_debug_logger(
    logger* logger);

/**
 * @brief Sets the logger for info, warning and error messages.
 * @details This logger will then be used for these messages.
 *
 * All logger objects which are no longer used for logging are deleted.
 * @param logger A pointer to the logger object to be used
 */
void
set_info_logger(
    logger* logger);

/**
 * @brief Sets the logger for warning and error messages.
 * @details This logger will then be used for these messages.
 *
 * All logger objects which are no longer used for logging are deleted.
 * @param logger A pointer to the logger object to be used
 */
void
set_warning_logger(
    logger* logger);

/**
 * @brief Sets the logger for error messages.
 * @details This logger will then be used for these messages.
 *
 * All logger objects which are no longer used for logging are deleted.
 * @param logger A pointer to the logger object to be used
 */
void
set_error_logger(
    logger* logger);

/**
 * @brief Sets all loggers to <tt>null</tt>.
 * @details Therefore, no messages will be logged and all loggers will be
 * deleted.
 */
void
clear_loggers();

/**
 * @brief Configures the logging system to process all logging messages.
 */
void
set_log_level_to_debug();

/**
 * @brief Configures the logging system to process info, warning and error
 * messages.
 */
void
set_log_level_to_info();

/**
 * @brief Configures the logging system to process warning and error messages.
 */
void
set_log_level_to_warning();

/**
 * @brief Configures the logging system to process error messages.
 */
void
set_log_level_to_error();

/**
 * @brief Logs a certain message as a debug message.
 * @details It will be logged if the log level is *debug* and the debug logger
 * is not <tt>null</tt>. If so, the message will be forwarded to this logger.
 * @param message The message to be logged
 * @see set_log_level_to_debug()
 * @see set_debug_logger(logger* logger)
 */
void
debug(
    std::string const& message);

/**
 * @brief Logs a certain message as an info message.
 * @details It will be logged if the log level is *debug* or *info* and the
 * info logger is not <tt>null</tt>. If so, the message will be forwarded to
 * this logger.
 * @param message The message to be logged
 * @see set_log_level_to_info()
 * @see set_info_logger(logger* logger)
 */
void
info(
    std::string const& message);

/**
 * @brief Logs a certain message as warning.
 * @details It will be logged if the log level is *debug*, *info* or *warning*
 * and the warning logger is not <tt>null</tt>. If so, the message will be
 * forwarded to this logger.
 * @param message The message to be logged
 * @see set_log_level_to_warning()
 * @see set_warning_logger(logger* logger)
 */
void
warning(
    std::string const& message);
  
/**
 * @brief Logs a certain message as an error message.
 * @details It will be logged if the error logger is not <tt>null</tt>. If so,
 * the message will be forwarded to this logger.
 * @param message The message to be logged
 * @see set_log_level_to_error()
 * @see set_error_logger(logger* logger)
 */
void
error(
    std::string const& message);

/**
 * @brief Checks if debug messages would currently be logged.
 * @returns <tt>true</tt> if debug messages would be logged
 */
bool
is_debug_on();

/**
 * @brief Checks if info messages would currently be logged.
 * @returns <tt>true</tt> if info messages would be logged
 */
bool
is_info_on();

/**
 * @brief Checks if warnings would currently be logged.
 * @returns <tt>true</tt> if warnings would be logged
 */
bool
is_warning_on();

/**
 * @brief Checks if error messages would currently be logged.
 * @returns <tt>true</tt> if error messages would be logged
 */
bool
is_error_on();

/**
 * @brief Logs the message *msg* as a debug message.
 * @details It will be logged if the log level is *debug* and the debug logger
 * is not <tt>null</tt>. If so, the message will be forwarded to this logger.
 * @see LOG_STREAM_DEBUG(msg)
 */
#define LOG_DEBUG(msg) mack::logging::debug(msg);

/**
 * @brief Logs the message *msg* as an info message.
 * @details It will be logged if the log level is *debug* or *info* and the
 * info logger is not <tt>null</tt>. If so, the message will be forwarded to
 * this logger.
 * @see LOG_STREAM_INFO(msg)
 */
#define LOG_INFO(msg) mack::logging::info(msg);

/**
 * @brief Logs the message *msg* as a warning.
 * @details It will be logged if the log level is *debug*, *info* or *warning*
 * and the warning logger is not <tt>null</tt>. If so, the message will be
 * forwarded to this logger.
 * @see LOG_STREAM_WARNING(msg)
 */
#define LOG_WARNING(msg) mack::logging::warning(msg);

/**
 * @brief Logs the message *msg* as an error message.
 * @details It will be logged if the error logger is not <tt>null</tt>. If so,
 * the message will be forwarded to this logger.
 * @see LOG_STREAM_ERROR(msg)
 */
#define LOG_ERROR(msg) mack::logging::error(msg);

/**
 * @brief Logs the message *msg* as a debug message.
 * @details It will be logged if the log level is *debug* and the debug logger
 * is not <tt>null</tt>. If so, the message will be forwarded to this logger.
 *
 * The message can be composed of multiple parts of different types which are
 * concatenated by the <tt>\<\<</tt> operator.
 * For example: <tt>LOG_STREAM_DEBUG("part 1, " << 2 << ", part 3");</tt>
 * @see LOG_DEBUG(msg)
 */
#define LOG_STREAM_DEBUG(msg) if (mack::logging::is_debug_on()) {std::stringstream ss;ss << msg;mack::logging::debug(ss.str());}

/**
 * @brief Logs the message *msg* as an info message.
 * @details It will be logged if the log level is *debug* or *info* and the
 * info logger is not <tt>null</tt>. If so, the message will be forwarded to
 * this logger.
 *
 * The message can be composed of multiple parts of different types which are
 * concatenated by the <tt>\<\<</tt> operator.
 * For example: <tt>LOG_STREAM_INFO("part 1, " << 2 << ", part 3");</tt>
 * @see LOG_INFO(msg)
 */
#define LOG_STREAM_INFO(msg) if (mack::logging::is_info_on()) {std::stringstream ss;ss << msg;mack::logging::info(ss.str());}

/**
 * @brief Logs the message *msg* as a warning.
 * @details It will be logged if the log level is *debug*, *info* or *warning*
 * and the warning logger is not <tt>null</tt>. If so, the message will be
 * forwarded to this logger.
 *
 * The message can be composed of multiple parts of different types which are
 * concatenated by the <tt>\<\<</tt> operator.
 * For example: <tt>LOG_STREAM_WARNING("part 1, " << 2 << ", part 3");</tt>
 * @see LOG_WARNING(msg)
 */
#define LOG_STREAM_WARNING(msg) if (mack::logging::is_warning_on()) {std::stringstream ss;ss << msg;mack::logging::warning(ss.str());}

/**
 * @brief Logs the message *msg* as an error message.
 * @details It will be logged if the error logger is not <tt>null</tt>. If so,
 * the message will be forwarded to this logger.
 *
 * The message can be composed of multiple parts of different types which are
 * concatenated by the <tt>\<\<</tt> operator.
 * For example: <tt>LOG_STREAM_ERROR("part 1, " << 2 << ", part 3");</tt>
 * @see LOG_ERROR(msg)
 */
#define LOG_STREAM_ERROR(msg) if (mack::logging::is_error_on()) {std::stringstream ss;ss << msg;mack::logging::error(ss.str());}


} // namespace logging
} // namespace mack

#endif /* __MACK_LOGGING_HPP__ */
