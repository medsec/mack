#ifndef __MACK_OPTIONS_OPTION_SWITCH_HPP__
#define __MACK_OPTIONS_OPTION_SWITCH_HPP__

#include <mack/options/selection_option.hpp>

#include <string>

namespace mack {
namespace options {

/**
 * @brief An option which can only be <tt>true</tt> or <tt>false</tt>.
 * @details The option is set to <tt>false</tt> initially and can be set to
 * <tt>true</tt> by using the <tt>set_value</tt> method. The
 * <tt>get_boolean</tt> method of the created \ref value object
 * (see \ref create()) can be employed to access the state.
 *
 * @author Johannes Kiesel
 * @date Aug 24 2012
 * @see set_value()
 * @see mack::options::option
 */
class option_switch : public selection_option {

  public:

    /**
     * @brief Create a new option switch.
     * @param short_flag the short flag (typically a single character) of this
     * option
     * @param long_flag the long flag (typically a word) of this option
     * @param brief_description a short description of this option
     * @param detailed_description a more detailed description (optional) or
     * the empty string
     */
    option_switch(
        std::string const& short_flag,
        std::string const& long_flag,
        std::string const& brief_description,
        std::string const& detailed_description);

    virtual
    ~option_switch();

  private:

    /**
     * @brief Creates a vector with the two entries "true" and "false".
     * @return the vector
     */
    static
    std::vector<std::string>
    create_true_false_vector()
    {
      std::vector<std::string> values;
      values.push_back("true");
      values.push_back("false");
      return values;
    }

}; // class option_switch

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_OPTION_SWITCH_HPP__ */
