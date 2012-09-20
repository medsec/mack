#ifndef __MACK_OPTIONS_OPTIONS_HPP__
#define __MACK_OPTIONS_OPTIONS_HPP__

#include <mack/options/option.hpp>
#include <mack/options/values.hpp>

#include <string>
#include <vector>

#include <set>
#include <map>

namespace mack {
namespace options {

/**
 * @brief The string for separating option namespaces.
 */
const std::string option_namespace_separator(".");

/**
 * @brief A set of options for program configuration.
 * @details The options are used as a configuration for either a program or an
 * option type.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see mack::options
 * @see mack::options::option
 * @see mack::options::program_options
 */
class options {

  public:

    /**
     * @brief Constructor for options.
     * @details The single options will be deleted if this object is deleted.
     * @param brief_description a brief description of the program or type for
     * which these options are used
     * @param detailed_description a detailed description
     * @param options the single options
     * @throws flag_collision_error if a flag is not unique within the options
     * @throws null_pointer_error if one of the option pointers is <tt>null</tt>
     */
    options(
        std::string const& brief_description,
        std::string const& detailed_description,
        std::vector<option*> const& options);

    /**
     * @brief The destructor.
     * @details The single options will be deleted.
     */
    virtual ~options();

    /**
     * @brief Gets the brief description of the program or type for which these
     * options are used.
     * @return the brief description
     */
    std::string const&
    get_brief_description() const;

    /**
     * @brief Gets the full description of the program or type for which these
     * options are used.
     * @return the full description
     */
    std::string const&
    get_full_description() const;

    /**
     * @brief Gets an iterator over all options.
     * @return the iterator
     */
    std::vector<option*>::iterator
    begin();

    /**
     * @brief Gets a const iterator over all options.
     * @return the iterator
     */
    std::vector<option const*>::const_iterator
    begin() const;

    /**
     * @brief Gets an iterator to the end of the options.
     * @return the iterator
     */
    std::vector<option*>::iterator
    end();

    /**
     * @brief Gets a const iterator to the end of the options.
     * @return the iterator
     */
    std::vector<option const*>::const_iterator
    end() const;

    /**
     * @brief Gets the option with the specified flag (short or long).
     * @details if there is no option with such a flag, an error is thrown.
     * @param flag the short or long flag of the desired option
     * @return a pointer to the option
     * @throws no_such_option_error if no option with such a flag exists
     */
    virtual
    option*
    get_option(
        std::string const& flag);

    /**
     * @brief Gets the option with the specified flag (short or long).
     * @details If there is no option with such a flag, an error is thrown.
     * @param flag the short or long flag of the desired option
     * @return a const pointer to the option
     * @throws no_such_option_error if no option with such a flag exists
     */
    virtual
    option const*
    get_option(
        std::string const& flag) const;

    /**
     * @brief Checks if this object has at least one child option.
     * @return if there is at least one child option
     */
    bool
    is_empty() const;

    /**
     * @brief Creates and returns the values of each child option.
     * @details Errors which occur while creating one of the values will not be
     * caught by this method. These errors are likely to be of type 
     * invalid_value_error or invalid_type_error.
     * @return a mapping of option long flag to the value created from the
     * corresponding option
     * @throws no_value_error if no value is set for an option.
     * @throws invalid_value_error if the value that was assigned to an option
     * is invalid for this option (e.g., a negative value assigned to an option
     * which only takes natural numbers)
     * @throws invalid_type_error if the value that was assigned to an option
     * is of an invalid type (e.g., a cracker assigned to an option which only
     * takes algorithms). This is most likely an undetected compile time error
     */
    virtual
    values const*
    create() const;

    /**
     * @brief Searches recursively for all options with the specified flag.
     * @param flag a short or long flag
     * @param this_namespace the option namespace of this options object
     * @return the found options as (options-namespace, option-object) pairs
     */
    virtual
    std::set<std::pair<std::string, option*> >
    search_for_option(
        std::string const& flag,
        std::string const& this_namespace);

  private:

    /**
     * @brief The default constructor.
     */
    options();
     

    /**
     * @brief The copy constructor.
     * @param original the values to copy from
     */
    options(
        options const& original);

    /**
     * @brief The assignment operator.
     * @param original the values to copy from
     * @return this object
     */
    options&
    operator=(
        options const& original);

    /**
     * @brief The brief description.
     */
    std::string _brief;

    /**
     * @brief The full description.
     */
    std::string _full;

    /**
     * @brief The child options.
     */
    std::vector<option*> _options;

    /**
     * @brief The child options as const pointers.
     */
    std::vector<option const*> _const_options;

}; // class options

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_OPTIONS_HPP__ */
