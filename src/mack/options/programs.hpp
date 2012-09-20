#ifndef __MACK_OPTIONS_PROGRAMS_HPP__
#define __MACK_OPTIONS_PROGRAMS_HPP__

#include <string>

#include <mack/options/program_options.hpp>

namespace mack {
namespace options {

/**
 * @namespace mack::options::programs
 * @brief A namespace which contains methods concerning *Mack the Knife*
 * programs.
 * @see create_program_option(std::string const& program_name)
 */
namespace programs {

/**
 * @brief An error which indicates that a program is not known.
 *
 * @author Johannes Kiesel
 * @date Jul 30 2012
 * @see errinfo_program_name the unknown name
 */
struct no_such_program_error : virtual std::exception, virtual boost::exception
{
};

/**
 * @brief An error information type for indicating the name of a program.
 */
typedef boost::error_info<struct tag_option_namespace, std::string>
  errinfo_program_name;

/**
 * @brief Creates and returns the \ref mack::options::program_options object
 * for a specific program.
 * @details The object is created as specified in the documentation of the
 * corresponding program.
 *
 * It may also be the case that an error is thrown on the creation of the
 * program options (likely to be one of \ref mack::options or
 * \ref mack::core::files).
 * @param program_name the name of the program as specified in the
 * <tt>\@program \<name\></tt> annotation
 * @return the corresponding program options
 * @throws no_such_program_error if no program with this name exists
 */
program_options*
create_program_option(
    std::string const& program_name);

} // namespace programs
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_PROGRAMS_HPP__ */

