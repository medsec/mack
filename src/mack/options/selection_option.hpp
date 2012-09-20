#ifndef __MACK_OPTIONS_SELECTION_OPTION_HPP__
#define __MACK_OPTIONS_SELECTION_OPTION_HPP__

#include <mack/options/option.hpp>

#include <string>
#include <vector>

namespace mack {
namespace options {


/**
 * @brief A program option to which only one of a set of predefined values can
 * be assigned.
 *
 * @author Johannes Kiesel
 * @date Aug 24 2012
 * @see mack::options::option
 */
class selection_option : public option {

  public:

    /**
     * @brief Create a new mandatory selection option.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     * @param selection_values the values that can be assigned to this option
     * @throws no_selection_error if no value to be selected was given
     */
    selection_option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        std::vector<std::string> const& selection_values);

    /**
     * @brief Create a new option with a default value.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     * @param selection_values the values that can be assigned to this option
     * @param default_value the value to be used if no other value is set
     * (e.g., after the construction)
     * @throws no_selection_error if no value to be selected was given
     * @throws invalid_value_error if the default value is not a valid value
     * in the sense of <tt>is_valid_value</tt>
     * @see is_valid_value(std::string const& value) const
     */
    selection_option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        std::vector<std::string> const& selection_values,
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
     * @param selection_values the values that can be assigned to this option
     * @param default_value the value to be used if no other value is set
     * @param start_value the initial value
     * @throws no_selection_error if no value to be selected was given
     * @throws invalid_value_error if the default value or the start value is
     * not a valid value in the sense of <tt>is_valid_value</tt>
     * @see is_valid_value(std::string const& value) const
     */
    selection_option(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description,
        std::vector<std::string> const& selection_values,
        std::string const& default_value,
        std::string const& start_value);

    virtual
    ~selection_option();

    virtual
    bool
    is_selection() const;

    virtual
    std::vector<std::string>::const_iterator
    selection_values_begin() const;

    virtual
    std::vector<std::string>::const_iterator
    selection_values_end() const;

  private:

    /**
     * @brief The possible values.
     */
    std::vector<std::string> _selection_values;

}; // class selection_option

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_SELECTION_OPTION_HPP__ */
