#ifndef __MACK_OPTIONS_EXCEPTIONS_HPP__
#define __MACK_OPTIONS_EXCEPTIONS_HPP__

#include <exception>
#include <boost/exception/exception.hpp>
#include <boost/exception/error_info.hpp>
#include <boost/exception/info.hpp>
#include <mack/core/files.hpp>
#include <set>

namespace mack {
namespace options {

/**
 * @brief An error which is thrown in order to indicate that an options flag is
 * used multiple times within an option namespace.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see errinfo_option_namespace the namespace of the options
 * @see errinfo_option_flag the flag which is used multiple times
 */
struct flag_collision_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which is thrown in order to indicate that an incorrect
 * flag is used. Flags can only contain letters, numbers, dashes and
 * underscores. The first character must not be a dash or underscore.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_namespace the namespace of the option
 * @see errinfo_option_flag the incorrect flag
 */
struct invalid_flag_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which is thrown in order to indicate that an %value of an
 * invalid type was supplied for an option.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see errinfo_option_type the desired type for the option
 * @see errinfo_option_value the %value assigned to the option (of invalid type)
 * @see errinfo_option_flag the flag of the option
 * @see errinfo_option_namespace the namespace of the option
 */
struct invalid_type_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which is thrown in order to indicate that an invalid %value
 * was supplied for an option.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see errinfo_option_value the invalid %value assigned to the option
 * @see errinfo_option_flag the flag of the option
 * @see errinfo_option_namespace the namespace of the option
 * @see errinfo_option_value_description a short description of what a correct
 * %value would have been. For example a pattern or an inequality (thus language
 * independent)
 */
struct invalid_value_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which is thrown in order to indicate that a %value
 * was supplied for a type option which could refer to multiple classes.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see errinfo_option_value the ambiguous %value assigned to the option
 * @see errinfo_option_value_senses the senses which could have be meant
 * @see errinfo_option_flag the flag of the option
 * @see errinfo_option_namespace the namespace of the option
 */
struct ambiguous_value_error : invalid_value_error
{
};

/**
 * @brief An error indicating that a %value is not bound to an option by
 * the use of an flag.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_value the unbound %value
 */
struct unbound_value_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which indicates that an option is not known.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_namespace the namespace of the unknown option
 * @see errinfo_option_flag the flag of the unknown option
 */
struct no_such_option_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which is thrown if an argument flag is ambiguous.
 * @details Option flags from different sub-namespaces are not
 * necessarily different. Nevertheless, they may be set by the user
 * without specifying the proper option namespaces. In this case, this
 * error will be thrown.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_names the options (flag including namespace)
 * that would be a valid choice for the ambiguous argument.
 * @see errinfo_option_flag the flag which is ambiguous
 * @see errinfo_option_value the %value which is assigned to the flag,
 * if any
 */
struct option_collision_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which indicates that a mandatory option has not been set
 * by the user.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_namespace the namespace that should have included
 * the missing %value
 * @see errinfo_option_flag the (long) flag of the missing option
 */
struct no_value_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which indicates that an option is not a selection, but
 * should be.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_namespace the namespace of the option
 * @see errinfo_option_flag the (long) flag of the option
 */
struct no_selection_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error which is thrown when it is queried for a certain
 * namespace which does not exist.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_option_namespace the parent namespace
 * @see errinfo_option_flag the flag of the namespace which is missing
 */
struct no_such_namespace_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error information type for indicating the
 * colliding options which caused an \ref option_collision_error.
 * @see option_collision_error
 */
typedef boost::error_info<struct tag_option_name_second, std::set<std::string> >
  errinfo_option_names;

/**
 * @brief An error information type for indicating an option flag.
 */
typedef boost::error_info<struct tag_option_flag, std::string>
  errinfo_option_flag;

/**
 * @brief An error information type for indicating an option %value.
 */
typedef boost::error_info<struct tag_option_value, std::string>
  errinfo_option_value;

/**
 * @brief An error information type for providing the possible senses
 * of an ambiguous value.
 * @see ambiguous_value_error
 */
typedef boost::error_info<struct tag_option_value,
        std::pair<std::string, std::string> >
  errinfo_option_value_senses;

/**
 * @brief An error information type for indicating what a valid value would
 * have been.
 * @see invalid_value_error
 */
typedef boost::error_info<struct tag_option_value_description, std::string>
  errinfo_option_value_description;

/**
 * @brief An error information type for indicating the desired type for an
 * option.
 */
typedef boost::error_info<struct tag_option_type, std::string>
  errinfo_option_type;

/**
 * @brief An error information type for indicating an option namespace.
 */
typedef boost::error_info<struct tag_option_namespace, std::string>
  errinfo_option_namespace;

/**
 * @brief Gets an error message for a flag_collision_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    flag_collision_error const& error);

/**
 * @brief Gets an error message for an unbound_value_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    unbound_value_error const& error);

/**
 * @brief Gets an error message for a no_selection_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    no_selection_error const& error);

/**
 * @brief Gets an error message for an invalid_flag_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    invalid_flag_error const& error);

/**
 * @brief Gets an error message for a no_such_namespace_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    no_such_namespace_error const& error);

/**
 * @brief Gets an error message for a no_such_option_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    no_such_option_error const& error);

/**
 * @brief Gets an error message for an option_collision_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    option_collision_error const& error);

/**
 * @brief Gets an error message for an invalid_value_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    invalid_value_error const& error);

/**
 * @brief Gets an error message for an ambiguous_value_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    ambiguous_value_error const& error);

/**
 * @brief Gets an error message for an invalid_type_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    invalid_type_error const& error);

/**
 * @brief Gets an error message for a no_value_error.
 * @param error the error
 * @return a message describing the error in a human readable form
 */
std::string
get_error_message(
    no_value_error const& error);

/**
 * @def CATCH_AND_GET_MESSAGE(code, message)
 * Executes <tt>code</tt> and catches the exceptions of \ref mack::core::files
 * and \ref mack::options. If an exception is caught, the execution of
 * <tt>code</tt> is interrupted and a human readable error message is written
 * to <tt>message</tt>.
 */
#define CATCH_AND_GET_MESSAGE(code, message)\
  try\
  {\
    code;\
  }\
  catch (mack::options::flag_collision_error const& e)\
  {\
    message = mack::options::get_error_message(e);\
  }\
  catch (mack::options::unbound_value_error const& e)\
  {\
    message = mack::options::get_error_message(e);\
  }\
  catch (mack::options::no_selection_error const& e)\
  {\
    message = mack::options::get_error_message(e);\
  }\
  catch (mack::options::invalid_flag_error const& e)\
  {\
    message = mack::options::get_error_message(e);\
  }\
  catch (mack::options::no_such_namespace_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::options::no_such_option_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::options::option_collision_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::options::ambiguous_value_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::options::invalid_value_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::options::invalid_type_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::options::no_value_error const& e)\
  {\
    message = ::mack::options::get_error_message(e);\
  }\
  catch (mack::core::files::file_not_exists_error const& e)\
  {\
    message = ::mack::core::files::get_error_message(e);\
  }\
  catch (mack::core::files::not_a_file_error const& e)\
  {\
    message = ::mack::core::files::get_error_message(e);\
  }\
  catch (mack::core::files::not_a_directory_error const& e)\
  {\
    message = ::mack::core::files::get_error_message(e);\
  }\
  catch (mack::core::files::file_read_error const& e)\
  {\
    message = ::mack::core::files::get_error_message(e);\
  }\
  catch (mack::core::files::file_write_error const& e)\
  {\
    message = ::mack::core::files::get_error_message(e);\
  }\
  catch (mack::core::files::parse_error const& e)\
  {\
    message = ::mack::core::files::get_error_message(e);\
  }

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_EXCEPTIONS_HPP__ */
