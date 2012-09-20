#ifndef __MACK_OPTIONS_HANDLERS_HANDLER_HPP__
#define __MACK_OPTIONS_HANDLERS_HANDLER_HPP__

#include <mack/options/option.hpp>
#include <mack/options/options.hpp>
#include <mack/options/program_options.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/files.hpp>

#include <string>
#include <vector>
#include <iostream>

/**
 * @option_type{option_handlers,Option Handlers}
 * @brief Different implementations for interactive option configuration.
 * @option_type_class{mack::options::handlers::handler}
 */

namespace mack {
namespace options {

/**
 * @namespace mack::options::handlers
 * @brief Namespace containing interaction handlers for *Mack the Knife*
 * programs.
 */
namespace handlers {

/**
 * @brief Base class of interaction handlers for programs.
 * @details Interaction handlers are employed to allow the user to specify
 * program options more interactively. An interaction handler can be
 * specified by a default program option at program start.
 *
 * @author Johannes Kiesel
 * @date Aug 27 2012
 * @see mack::options
 */
class handler {

  public:

    /**
     * @brief Constructor for interaction handlers.
     * @details Note that the program options which shall be configured with
     * this handler have also to be set.
     * @see set_program_options(program_options* options)
     */
    handler();

    /**
     * @brief The destructor.
     */
    virtual
    ~handler();

    /**
     * @brief Sets the program options.
     * @details These program options will then be configured by using the
     * <tt>run</tt> method of this handler.
     * @param options the program options
     * @throws null_pointer_error it given options are <tt>null</tt>
     * @see run()
     */
    void
    set_program_options(
        program_options* options);

    /**
     * @brief Uses this interaction handler to configure the program options
     * and create the specified values.
     * @return the created values
     * @throws ::mack::options::exit_requested if program exit was requested
     */
    virtual
    values const*
    run() = 0;

  protected:

    /**
     * @brief Gets the name of the program.
     * @return the program name
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    std::string const&
    get_program_name() const;

    /**
     * @brief Gets a brief description of the program.
     * @return the description
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    std::string const&
    get_brief_program_description() const;

    /**
     * @brief Gets a full description of the program.
     * @return the description
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    std::string const&
    get_full_program_description() const;

    /**
     * @brief Gets the configurable options of the program.
     * @return the options
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    std::vector<option const*>
    get_options() const;

    /**
     * @brief Sets a program option to the default value of the option (if any).
     * @param flag the flag (long or short, including option namespace or not)
     * of the program option
     * @param[out] was_successfull if setting the option was successful
     * @param[out] error_message a human readable error message if setting
     * the option was not successful
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    void
    set(
        std::string const& flag,
        bool& was_successfull,
        std::string& error_message);

    /**
     * @brief Sets a program option to a specific value.
     * @param flag the flag (long or short, including option namespace or not)
     * of the program option
     * @param value the value
     * @param[out] was_successfull if setting the option was successful
     * @param[out] error_message a human readable error message if setting
     * the option was not successful
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    void
    set(
        std::string const& flag,
        std::string const& value,
        bool& was_successfull,
        std::string& error_message);

    /**
     * @brief Creates the values from the current program options configuration.
     * @param[out] was_successfull if the creation was successful
     * @param[out] error_message a human readable error message if the
     * creation was not successful
     * @throws null_pointer_error if the program options are not yet set
     * @see set_program_options(program_options* options)
     */
    values const*
    create(
        bool& was_successfull,
        std::string& error_message) const;

  private:

    /**
     * @brief Checks it <tt>_options</tt> is <tt>null</tt> and returns it if not.
     * @return <tt>_options</tt>
     * @throws ::mack::core::null_pointer_error
     * if <tt>_options</tt> is <tt>null</tt>
     */
    program_options*
    options();

    /**
     * @brief Checks it <tt>_options</tt> is <tt>null</tt> and returns it if not.
     * @return <tt>_options</tt>
     * @throws ::mack::core::null_pointer_error
     * if <tt>_options</tt> is <tt>null</tt>
     */
    program_options const*
    options() const;

    program_options* _options;

}; // class handler

} // namespace handlers
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_HANDLERS_HANDLER_HPP__ */

