#ifndef __MACK_OPTIONS_OPTION_TYPE_HPP__
#define __MACK_OPTIONS_OPTION_TYPE_HPP__

#include <mack/options/value.hpp>
#include <mack/options/values.hpp>
#include <mack/options/options.hpp>

#include <string>

namespace mack {
namespace options {

/**
 * @brief Base class for option types.
 * @details For each option type, a concrete option type class will be created,
 * compiled and linked. The concrete class <i>typename</i> can be accessed by
 * including <tt>mack/options/types/<i>typename</i>.hpp</tt>, with
 * <i>typename</i> being the name of the type as specified in the
 * <tt>\@option_type</tt> annotation.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see mack::options
 */
class option_type {

  public:

    /**
     * @brief The constructor.
     */
    option_type();

    /**
     * @brief The destructor.
     */
    virtual
    ~option_type();

    /**
     * @brief Get the names (including namespace) of all classes which are of
     * this type.
     * @return a vector containing all class names
     */
    virtual
    std::vector<std::string> const&
    get_class_names() const = 0;

    /**
     * @brief Get the options for a certain class of this type.
     * @details If there are no options for the specified class, an empty
     * options object is returned.
     * @param class_name the name of the class
     * @return the options
     * @throws invalid_value_error if this type has no member class with given
     * class name
     */
    virtual
    options*
    get_options(
        std::string const& class_name) const = 0;

    /**
     * @brief Creates an instance of a class of this type.
     * @details Errors which occur while creating the value will not be
     * caught by this method. These errors should be of type
     * invalid_value_error or invalid_type_error.
     * @param class_name the name of the class
     * @param values the values which were assigned to the options of the class
     * (see get_options(std::string const& class_name))
     * @param is_template_value if the created value is then used as a template
     * value.
     * @return the created value
     * @throws no_value_error if no value is set for an option. This is most
     * likely an undetected compile time error
     * @throws invalid_value_error if this type has no member class with given
     * class name. Generally, if the value that was assigned to an option
     * is invalid for this option (e.g., a negative value assigned to an option
     * which only takes natural numbers)
     * @throws invalid_type_error if the value that was assigned to an option
     * is of an invalid type (e.g., a cracker assigned to an option which only
     * takes algorithms). This is most likely an undetected compile time error
     */
    virtual
    value*
    create_value(
        std::string const& class_name,
        values const* values,
        const bool is_template_value) const = 0;

}; // class option_type

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_OPTION_TYPE_HPP__ */

