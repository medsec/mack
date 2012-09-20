#ifndef __MACK_OPTIONS_TYPE_OPTION_HPP__
#define __MACK_OPTIONS_TYPE_OPTION_HPP__

#include <mack/options/option.hpp>
#include <mack/options/option_type.hpp>
#include <mack/options/value.hpp>

#include <string>

namespace mack {
namespace options {


/**
 * @brief An option for configuring and creating an instance of classes of a
 * certain \ref option_type.
 *
 * @author Johannes Kiesel
 * @date Aug 24 2012
 * @see mack::options
 * @see mack::options::option_type
 */
class type_option : public option {

  public:

    /**
     * @brief Create a new mandatory option.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     * @param is_for_template if this option is used as a template parameter.
     * @param type the type of classes which can be assigned to this option.
     * This type instance will be deleted if this type_option instance is
     * deleted
     * @throws null_pointer_error if given type is <tt>null</tt>
     */
    type_option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        const bool is_for_template,
        option_type const* type);

    /**
     * @brief Create a new option with a default value.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     * @param is_for_template if this option is used as a template parameter.
     * @param type the type of classes which can be assigned to this option.
     * This type instance will be deleted if this type_option instance is
     * deleted
     * @param default_value the value to be used if no other value is set
     * (e.g., after the construction)
     * @throws null_pointer_error if given type is <tt>null</tt>
     * @throws invalid_value_error if the default value is not a valid value
     * in the sense of <tt>is_valid_value</tt>
     * @see is_valid_value(std::string const& value) const
     */
    type_option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        const bool is_for_template,
        option_type const* type,
        std::string const& default_value);

    /**
     * @brief Create a new option with a default value and a different
     * initial value.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     * @param is_for_template if this option is used as a template parameter.
     * @param type the type of classes which can be assigned to this option.
     * This type instance will be deleted if this type_option instance is
     * deleted
     * @param default_value the value to be used if no other value is set
     * @param start_value the initial value
     * @throws null_pointer_error if given type is <tt>null</tt>
     * @throws invalid_value_error if the default value or the start value is
     * not a valid value in the sense of <tt>is_valid_value</tt>
     * @see is_valid_value(std::string const& value) const
     */
    type_option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        const bool is_for_template,
        option_type const* type,
        std::string const& default_value,
        std::string const& start_value);

    virtual
    ~type_option();

    virtual
    value const*
    create() const;

    virtual
    bool
    has_child_options() const;

    virtual
    bool
    is_selection() const;

    virtual
    std::vector<std::string>::const_iterator
    selection_values_begin() const;

    virtual
    std::vector<std::string>::const_iterator
    selection_values_end() const;

    virtual
    bool
    is_valid_value(
        std::string const& value) const;

    virtual
    void
    set_value();

    virtual
    void
    set_value(
        std::string const& value);

    virtual
    options const*
    get_child_options() const;

    virtual
    options*
    get_child_options();

    virtual
    option const*
    get_child_option(
        std::string const& flag) const;

    virtual
    option*
    get_child_option(
        std::string const& flag);

  private:

    /**
     * @brief Deletes current options.
     */
    void
    clear_options();

    /**
     * @brief Gets the complete class name (with namespace) from a class name
     * (possible without namespace).
     * @details This method will compare given name (%value) to all known class
     * names. It will use different matching methods in this order:
     * <ol>
     * <li>Checking for a complete match</li>
     * <li>Checking for a complete but case insensitive match</li>
     * <li>Checking for a match with a simple class name
     * (without namespace)</li>
     * <li>Checking for a case insensitive match with a simple class name
     * (without namespace)</li>
     * </ol>
     * If a match is found for one method, the remaining methods will not be
     * tested. If multiple matches are found for one method, an error is
     * thrown. Similarly, an error is thrown if no match is found at all.
     * @param value the class name
     * @return the complete class name
     * @throws ambiguous_value_error if multiple matches are found within one
     * round
     * @throws invalid_value_error if no match is found at all
     */
    std::string
    get_class_name(
        std::string const& value) const;

    /**
     * @brief The option type of the classes which can be created from this
     * option.
     */
    option_type const* _type;

    /*
     * @brief The child options of the class currently assigned to this
     * option.
     */
    options* _options;

    bool _is_for_template;

}; // class type_option

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_TYPE_OPTION_HPP__ */
