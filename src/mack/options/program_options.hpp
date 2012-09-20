#ifndef __MACK_OPTIONS_PROGRAM_OPTIONS_HPP__
#define __MACK_OPTIONS_PROGRAM_OPTIONS_HPP__

#include <mack/options/options.hpp>

#include <boost/filesystem/path.hpp>

#include <string>
#include <iostream>

namespace mack {
namespace options {

namespace handlers {
class handler; // forward declaration
}

/**
 * @brief A set of options for a single program.
 * @details Instances of this class are used as configurations for a specific
 * *Mack the Knife* program. Each such program has a set of predefined default
 * options (e.g., <tt>--help</tt>) and specific options as well.
 *
 * @author Johannes Kiesel
 * @date Aug 27 2012
 * @see mack::options
 * @see mack::options::options
 */
class program_options : public options {

  public:

    /**
     * @brief Creates a set of options for a single program.
     * @param name the program name
     * @param brief_description a short description of the program
     * @param detailed_description a more detailed description of the program
     * @param options the specific options of this program
     */
    program_options(
        std::string const& name,
        std::string const& brief_description,
        std::string const& detailed_description,
        std::vector<option*> const& options);

    virtual
    ~program_options();

    /**
     * @brief Gets the name of this program.
     * @return the name
     */
    std::string const&
    get_program_name() const;

    /**
     * @brief Resets an option to its default value (if any).
     * @param flag the flag (long or short,
     * including the option namespace or not) of the option
     * @throws invalid_flag_error if given flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if the namespace does not exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws no_value_error if the option is the configuration file
     * option, which requires a value at this step
     */
    void
    set(
        std::string const& flag);

    /**
     * @brief Resets an option to its default value (if any).
     * @param namespaces the option namespaces of the option; separated as
     * single elements instead of by the \ref option_namespace_separator
     * @param flag the flag (long or short) of the option
     * @throws invalid_flag_error if given flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if the namespace does not exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws no_value_error if the option is the configuration file
     * option, which requires a value at this step
     */
    void
    set(
        std::vector<std::string> const& namespaces,
        std::string const& flag);

    /**
     * @brief Assigns a value to an option.
     * @details Note that values are not be assigned to the configuration
     * file option. Instead, the value is treated as path of a configuration
     * file, which is parsed. The parsed settings are then included into this
     * object.
     * @param flag the flag (long or short,
     * including the option namespace or not) of the option
     * @param value the value which should be assigned to the option
     * @throws invalid_flag_error if given flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if the namespace does not exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws invalid_value_error if the specified value is not a valid one
     * for the specified option
     * @throws ::mack::core::files::file_not_exists_error
     * if the option is the configuration file
     * option, but the path specified by the value does not exist
     * @throws ::mack::core::files::not_a_file_error
     * if the option is the configuration file
     * option, but the path specified by the value is not a file
     * @throws ::mack::core::files::file_read_error
     * if the option is the configuration file
     * option, but an error occured while reading the file which is specified
     * by the value
     */
    void
    set(
        std::string const& flag,
        std::string const& value);

    /**
     * @brief Assigns a value to an option.
     * @details Note that values are not be assigned to the configuration
     * file option. Instead, the value is treated as path of a configuration
     * file, which is parsed. The parsed settings are then included into this
     * object.
     * @param namespaces the option namespaces of the option; separated as
     * single elements instead of by the \ref option_namespace_separator
     * @param flag the flag (long or short) of the option
     * @param value the value which should be assigned to the option
     * @throws invalid_flag_error if given flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if the namespace does not exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws invalid_value_error if the specified value is not a valid one
     * for the specified option
     * @throws ::mack::core::files::file_not_exists_error if the option is
     * the configuration file
     * option, but the path specified by the value does not exist
     * @throws ::mack::core::files::not_a_file_error if the option is the
     * configuration file
     * option, but the path specified by the value is not a file
     * @throws ::mack::core::files::file_read_error if the option is the
     * configuration file
     * option, but an error occured while reading the file which is specified
     * by the value
     */
    void
    set(
        std::vector<std::string> const& namespaces,
        std::string const& flag,
        std::string const& value);

    /**
     * @brief Parses and interprets the content of a configuration file.
     * @param configuration the content of the configuration file
     * @throws invalid_flag_error if a parsed flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if a parsed option namespace does not
     * exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws invalid_value_error if a parsed value is not a valid one
     * for the specified option
     * @throws ::mack::core::files::file_not_exists_error if the configuration
     * file
     * option is referenced, but the path specified by the value does not exist
     * @throws ::mack::core::files::not_a_file_error if the configuration file
     * option is referenced, but the path specified by the value is not a file
     * @throws ::mack::core::files::file_read_error if the configuration file
     * option is referenced, but an error occured while reading the file which
     * is specified by the value
     * @throws no_value_error if the the configuration file
     * option is referenced, which requires a value at this step,
     * but no value is assigned to it
     * @see mack::options::configuration_file
     */
    void
    set_from_configuration(
        std::string const& configuration);

    /**
     * @brief Parses and interprets a configuration file.
     * @param file_path the path of the configuration file
     * @throws invalid_flag_error if a parsed flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if a parsed option namespace does not
     * exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws invalid_value_error if a parsed value is not a valid one
     * for the specified option
     * @throws ::mack::core::files::file_not_exists_error if the specified path
     * does not exists or the configuration file
     * option is used again, but the path specified by the value does not exist
     * @throws ::mack::core::files::not_a_file_error if the specified path
     * does not point to a file or the configuration file
     * option is used again, but the path specified by the value is not a file
     * @throws ::mack::core::files::file_read_error if an error occurred while
     * reading a configuration file
     * @throws no_value_error if the the configuration file
     * option is referenced, which requires a value at this step,
     * but no value is assigned to it
     * @see mack::options::configuration_file
     */
    void
    set_from_configuration_file(
        boost::filesystem::path const& file_path);

    /**
     * @brief Configures these options from command line arguments.
     * @param argc the number of argv parameters
     * @param argv the parameters as an array of c strings
     * @param offset the first parameter of argv which should be parsed
     * @throws unbound_value_error if a value is encountered which is not
     * assigned to an option
     * @throws invalid_flag_error if a parsed flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if a parsed option namespace does not
     * exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     * @throws invalid_value_error if a parsed value is not a valid one
     * for the specified option
     * @throws ::mack::core::files::file_not_exists_error if the configuration
     * file
     * option is referenced, but the path specified by the value does not exist
     * @throws ::mack::core::files::not_a_file_error if the configuration file
     * option is referenced, but the path specified by the value is not a file
     * @throws ::mack::core::files::file_read_error if the configuration file
     * option is referenced, but an error occured while reading the file which
     * is specified by the value
     * @throws no_value_error if the the configuration file
     * option is referenced, which requires a value at this step,
     * but no value is assigned to it
     * @see mack::options::configuration_file
     */
    void
    set_all(
        const int argc,
        char const* const* argv,
        const int offset = 1);

    /**
     * @brief Fixes the values of the help and interaction handler options.
     */
    void
    set_commandline_arguments_done();

    virtual
    option*
    get_option(
        std::string const& flag);

    virtual
    option const*
    get_option(
        std::string const& flag) const;

    virtual
    values const*
    create() const;

    /**
     * @brief Checks if the help option is set to <tt>true</tt>.
     * @return the boolean value of the help option
     */
    bool
    is_help_set() const;

    /**
     * @brief Prints help on this program to an output stream.
     * @param output_stream the stream to print to
     */
    void
    print_help(
        std::ostream& output_stream = std::cout) const;

    /**
     * @brief Creates a interaction handler from the current configuration.
     * @return a pointer to the created handler
     */
    mack::options::handlers::handler*
    create_interaction_handler() const;

    /**
     * @brief Gets pointers to the default options,
     * which each program has in common.
     * @return the default options
     */
    std::vector<option const*>
    get_default_options() const;

    /**
     * @brief Gets pointers to the default options,
     * which each program has in common.
     * @return the default options
     */
    std::vector<option*>
    get_default_options();

    /**
     * @brief Gets the option for specifying configuration files.
     * @return the option
     */
    option const*
    get_configuration_file_option() const;

    /**
     * @brief Gets the option for specifying the debug logger.
     * @return the option
     * @see mack::logging
     */
    option const*
    get_debug_logger_option() const;

    /**
     * @brief Gets the option for specifying the info logger.
     * @return the option
     * @see mack::logging
     */
    option const*
    get_info_logger_option() const;

    /**
     * @brief Gets the option for specifying the warning logger.
     * @return the option
     * @see mack::logging
     */
    option const*
    get_warning_logger_option() const;

    /**
     * @brief Gets the option for specifying the error logger.
     * @return the option
     * @see mack::logging
     */
    option const*
    get_error_logger_option() const;

    /**
     * @brief Gets the option for specifying the logging verbosity.
     * @return the option
     * @see mack::logging
     */
    option const*
    get_verbosity_option() const;

    virtual
    std::set<std::pair<std::string, option*> >
    search_for_option(
        std::string const& flag,
        std::string const& this_namespace);

  private:

    static
    option*
    create_help_option();

    static
    option*
    create_configuration_file_option();

    static
    option*
    create_debug_logger_option();

    static
    option*
    create_info_logger_option();

    static
    option*
    create_warning_logger_option();

    static
    option*
    create_error_logger_option();

    static
    option*
    create_verbosity_option();

    static
    option*
    create_interaction_handler_option();

    void
    check_valid_flag(
        std::string const& flag) const;

    std::vector<std::string>
    separate_namespaces(
        std::string& flag) const;

    /**
     * @brief Retrieves an option by namespace and flag.
     * @param namespaces the option namespaces of the option; separated as
     * single elements instead of by the \ref option_namespace_separator
     * @param flag the flag (long or short) of the option
     * @throws invalid_flag_error if given flag is not a valid flag. A valid
     * flag consists of letters and digits at any position and dashes and
     * underscored at any but the first position
     * @throws no_such_namespace_error if the namespace does not exist
     * @throws no_such_option_error if an option with the specified flag does
     * not exist in the option namespace
     * @throws option_collision_error if no namespace is specified, no option
     * with the specified flag exists in the default namespace, but multiple
     * options with the specified flag exist in other namespaces
     */
    option*
    find_option(
        std::vector<std::string> const& namespaces,
        std::string const& flag);

    std::string _name;

    option* _help_option;

    option* _configuration_file_option;

    option* _debug_logger_option;

    option* _info_logger_option;

    option* _warning_logger_option;

    option* _error_logger_option;

    option* _verbosity_option;

    option* _interaction_handler_option;

    bool _commandline_arguments_done;

}; // class program_options

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_PROGRAM_OPTIONS_HPP__ */

