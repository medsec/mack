/**
 * \page page_logging The Build-in Logging Facility
 * \tableofcontents
 * *Mack the Knife* offers a build-in facility for performing logging while
 * program execution. It is configured through default program options which
 * define what messages to log and how to write the messages.
 *
 * \section page_logging_overview Overview and Configuration
 * There are 4 log levels defined for messages:
 *    * *Debug messages* are fine grained messages useful for tracking bugs.
 *      Such messages are not displayed by default.
 *    * *Info messages* provide useful information to the user.
 *      The current status of a long procedure is one example of an info
 *      message.
 *    * *Warnings* are logged if unusual behaviour is detected which might
 *      cause erroneous results.
 *    * *Error messages* are logged if something has definitely gone wrong.
 *      These messages should provide information on what has gone wrong and
 *      how to fix it.
 *
 * The log levels are ordered in the above mentioned order. Thus, if *warnings*
 * are logged, *error messages* are always also logged. However, the program
 * can be configured to log only *error messages*. This is done by using the
 * <tt>verbosity</tt> (<tt>v</tt>) option (e.g., <tt>-\-verbosity=warning</tt>).
 * By default, the log level is set to *info*.
 *
 * For each log level, a logger (see \ref loggers) can be specified, which
 * defines how the messages are handled. As a special case, the
 * \ref mack::logging::previous_logger can be specified in order to use the
 * logger of the next lower log level instead. By default the
 * <tt>debug_logger</tt> (<tt>dL</tt>) and the <tt>info_logger</tt>
 * (<tt>iL</tt>) log to standard output, while the
 * <tt>warning_logger</tt> (<tt>wL</tt>) and the
 * <tt>error_logger</tt> (<tt>eL</tt>) log to standard error.
 *
 * \section page_logging_usage Usage
 * In order to log messages, the file <tt>mack/logging.hpp</tt> has to be
 * included. This will provide the macros <tt>LOG_DEBUG(msg)</tt>,
 * <tt>LOG_INFO(msg)</tt>, <tt>LOG_WARNING(msg)</tt> and
 * <tt>LOG_ERROR(msg)</tt> which can be used to log string messages
 * (<tt>msg</tt>) with the corresponding log levels. For example,
 * <tt>LOG_INFO("an info message");</tt> can be used to log the
 * message <i>an info message</i> with the log level *info*.
 *
 * For convenient logging of messages which contain variables, the
 * <tt>LOG_STREAM_DEBUG(msg)</tt>, ..., <tt>LOG_STREAM_ERROR(msg)</tt> macros
 * are provided also.
 * These macros employ stringstreams in order to convert basic types
 * to strings. For example,
 * <tt>LOG_STREAM_WARNING("x is " << x << ", which is likely to be bad");</tt>
 * would log the message <i>x is 42, which is likely to be bad</i> if the
 * variable <tt>x</tt> is an integer and 42 is assigned to it.
 *
 * Note that, for all macros, the message is only composed if it would be
 * logged according to the <tt>verbosity</tt> option.
 *
 */

