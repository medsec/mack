#ifndef __MACK_OPTIONS_OPTION_HPP__
#define __MACK_OPTIONS_OPTION_HPP__

#include <mack/options/value.hpp>

#include <string>
#include <vector>

namespace mack {
namespace options {

class options; // forward declaration

/**
 * @brief A program option.
 * @details This class describes program options, including their <i>flags</i>,
 * their <i>descriptions</i> and their behaviour on the assignment of values
 * to them. Although methods for handling a predefined set of possible values
 * (selection) are already defined, this class has to be extended in order to
 * define such a selection. Similarly, this class as such is not capable of
 * handling <i>type options</i>.
 *
 * @author Johannes Kiesel
 * @date Aug 09 2012
 * @see mack::options
 * @see mack::options::selection_option
 * @see mack::options::type_option
 */
class option {

  public:

    /**
     * @brief Create a new mandatory option.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     */
    option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description);

    /**
     * @brief Create a new option with a default value.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     * @param default_value the value to be used if no other value is set
     * (e.g., after the construction)
     * @throws invalid_value_error if the default value is not a valid value
     * in the sense of <tt>is_valid_value</tt>
     * @see is_valid_value(std::string const& value) const
     */
    option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
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
     * @param default_value the value to be used if no other value is set
     * @param start_value the initial value
     * @throws invalid_value_error if the default value or the start value is
     * not a valid value in the sense of <tt>is_valid_value</tt>
     * @see is_valid_value(std::string const& value) const
     */
    option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        std::string const& default_value,
        std::string const& start_value);

    /**
     * @brief The destructor.
     */
    virtual ~option();

    /**
     * @brief Gets the short flag of this option.
     * @return the flag
     */
    std::string const&
    get_short_flag() const;

    /**
     * @brief Gets the long flag of this option.
     * @return the flag
     */
    std::string const&
    get_long_flag() const;

    /**
     * @brief Gets the brief description of this option.
     * @return the description
     */
    std::string const&
    get_brief_description() const;

    /**
     * @brief Gets the full description of this option.
     * @details The full description consists of brief and detailed
     * description.
     * @return the description
     */
    std::string const&
    get_full_description() const;

    /**
     * @brief Checks if a value is set for this option.
     * @return if a value is set
     */
    bool
    has_value() const;

    /**
     * @brief Gets the value which is currently assigned to this option.
     * @details If no value is set (<tt>has_value()</tt> is <tt>false</tt>),
     * an error will be thrown.
     * @return the current value
     * @throws no_value_error if no value is set
     * @see has_value() const
     */
    std::string
    get_value() const;

    /**
     * @brief Creates a value object from this option.
     * @details If no value is set (<tt>has_value()</tt> is <tt>false</tt>),
     * an error will be thrown.
     *
     * Note that if this is a <tt>type_option</tt>, calling this method will
     * result in creating a new object using the child options of this
     * option. Errors which occur while creating this object will not be
     * caught by this method. These errors are likely to be of type 
     * no_value_error, invalid_value_error or invalid_type_error.
     *
     * @return the current value as a value object
     * @throws no_value_error if no value is set
     * @see mack::options::type_option
     * @see mack::options::no_value_error
     * @see mack::options::invalid_type_error
     * @see mack::options::invalid_value_error
     * @see has_value() const
     * @see get_value() const
     */
    virtual
    value const*
    create() const; // throws if !has_value, error in constructor

    /**
     * @brief Check if this option has child options.
     * @details This method will return <tt>true</tt> if this option is a
     * <i>type option</i>.
     * @return if there are child options
     * @see mack::options::type_option
     */
    virtual
    bool
    has_child_options() const;

    /**
     * @brief Gets the child options of this option.
     * @details If there are no child options, an error will be thrown.
     * @return the child options
     * @throws no_such_namespace_error if there are no child options (and thus,
     * this is not a namespace)
     * @see has_child_options const
     */
    virtual
    options const*
    get_child_options() const;

    /**
     * @brief Gets the child options of this option.
     * @details If there are no child options, an error will be thrown.
     * @return the child options
     * @throws no_such_namespace_error if there are no child option (and thus,
     * this is not a namespace)
     * @see has_child_options const
     */
    virtual
    options*
    get_child_options();

    /**
     * @brief Gets the child option of this option with a certain flag.
     * @details If there are no child options or an option with this flag does
     * not exist, an error will be thrown.
     * @return the child option with given long or short flag
     * @throws no_such_namespace_error if there are no child option (and thus,
     * this is not a namespace)
     * @throws no_such_option_error if there is no option with given flag
     * @see has_child_options const
     */
    option const*
    get_child_option(
        std::string const& flag) const;

    /**
     * @brief Gets the child option of this option with a certain flag.
     * @details If there are no child options or an option with this flag does
     * not exist, an error will be thrown.
     * @return the child option with given long or short flag
     * @throws no_such_namespace_error if there are no child option (and thus,
     * this is not a namespace)
     * @throws no_such_option_error if there is no option with given flag
     * @see has_child_options const
     */
    option*
    get_child_option(
        std::string const& flag);

    /**
     * @brief Checks if only a limited number of values are valid for this
     * option.
     * @return if this is a selection
     */
    virtual
    bool
    is_selection() const;

    /**
     * @brief Gets an iterator over all possible values which points to the
     * first value.
     * @details If this is not a selection, an error will be thrown.
     * @return the iterator
     * @throws no_selection_error if this is not a selection
     * @see is_selection() const
     */
    virtual
    std::vector<std::string>::const_iterator
    selection_values_begin() const;

    /**
     * @brief Gets an iterator over all possible values which points to the
     * end.
     * @details If this is not a selection, an error will be thrown.
     * @return the iterator
     * @throws no_selection_error if this is not a selection
     * @see is_selection() const
     */
    virtual
    std::vector<std::string>::const_iterator
    selection_values_end() const;

    /**
     * @brief Checks if a given value is valid in the sense of a selection of
     * values.
     * @details This method can not guarantee that the value is valid if this
     * is not a selection.
     * @param value the value to be checked
     * @return if the value matches one of the values of the selection values
     */
    virtual
    bool
    is_valid_value(
        std::string const& value) const;

    /**
     * @brief Sets this option back to a default value.
     * @details If this option has no default value, any existing value will
     * simply be deleted instead. Thus <tt>has_value</tt> will then be
     * <tt>false</tt>
     * @see has_value() const
     */
    virtual
    void
    set_value();

    /**
     * @brief Sets this option.
     * @details If the value is not valid in the sense of
     * <tt>is_valid_value</tt>, an error will be thrown.
     * @param value the value to be assigned to this option
     * @throws invalid_value_error if the value is not valid
     * @see is_valid_value(std::string const& value) const
     */
    virtual
    void
    set_value(
        std::string const& value);

    /**
     * @brief Checks if a default value exists for this option.
     * @return if a value exists
     */
    bool
    has_default_value() const;

    /**
     * @brief Gets the default value of this option.
     * @return the default value
     * @throws no_value_error if no default value exists
     * @see has_defaul_value() const
     */
    std::string const&
    get_default_value() const; // throws if !has_default_value

  protected:

    /**
     * @brief Sets the default value of this option.
     * @details The value will <b>not</b> be assigned to the option.
     *
     * If the value is not valid in the sense of
     * <tt>is_valid_value</tt>, an error will be thrown.
     * @param default_value the new default value for this option
     * @throws invalid_value_error if the value is not valid
     * @see is_valid_value(std::string const& value) const
     */
    void
    set_default_value(
        std::string const& default_value);

  private:

    /**
     * @brief Create full description.
     * @param brief_description the brief description
     * @param detailed_description the detailed description
     * @return the full description
     */
    static
    std::string
    create_full_description(
        std::string const& brief_description,
        std::string const& detailed_description);

    /**
     * @brief The default constructor.
     */
    option();
     

    /**
     * @brief The copy constructor.
     * @param original the values to copy from
     */
    option(
        option const& original);

    /**
     * @brief The assignment operator.
     * @param original the values to copy from
     * @return this object
     */
    option&
    operator=(
        option const& original);

    /**
     * @brief The short flag of this option. Typically a character.
     */
    std::string _short_flag;

    /**
     * @brief The long flag of this option. Typically a word.
     */
    std::string _long_flag;

    /**
     * @brief A brief description of this option.
     */
    std::string _brief;

    /**
     * @brief A complete description of this option (brief + detailed).
     */
    std::string _full;

    /**
     * @brief If this option has a default value.
     */
    bool _has_default_value;

    /**
     * @brief The default value of this option, if any.
     * @see _has_default_value
     */
    std::string _default_value;

    /**
     * @brief If this option has a value currently assigned to it.
     */
    bool _has_value;

    /**
     * @brief The value which is currently assigned to this option.
     * @see _has_value
     */
    std::string _value;

}; // class option

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_OPTION_HPP__ */
