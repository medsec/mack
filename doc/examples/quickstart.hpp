#include <mack/options/values.hpp>
#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>

namespace my_namespace {

/**
 * ...
 * @is_of_type{my_class_type}
 *
 * @option_switch{s,switch}
 * An option that takes no value.
 *
 * @option{d,default_value_option,default_value}
 * If this option is not set by the user, its value will
 * be set to *default_value*.
 *
 * @option{m,mandatory_option}
 * This option must be set by the user. It has to be a
 * non-negative integer value. This is checked in the
 * constructor.
 *
 * @type_option{D,default_type_option,my_class_type,my_namespace::my_extended_class}
 * This option can take any class which
 * *\@is_of_type{my_class_type}* and defaults to the class
 * *my_namespace::my_extended_class*.
 *
 * @type_option{M,mandatory_type_option,my_class_type}
 * This option is mandatory, since it has no default value
 * assigned. Otherwise like the *default_type_option*.
 *
 * @template_option{OPTIONAL,o,optional_template,my_no_class_type,my_namespace::my_other_class}
 * This option will set the *OPTIONAL* template parameter
 * and defaults to *my_namespace::my_other_class*. Note
 * that template options can also be of types that have no
 * *\@option_type_class* assigned to them.
 *
 * @template_option{MANDATORY,x,mandatory_template,my_no_class_type}
 * Of course, there is also a variant without a default
 * value (which is then mandatory).
 */
template<class MANDATORY, class OPTIONAL>
class my_class : public my_class_type_class
{
 public:
  /**
   * @brief The program options constructor.
   * @details If the class has some options defined, a
   * constructor with this signature
   * (i.e., which takes a *mack::options::values* const
   * pointer as its only input.) will be used.
   * Otherwise, the default constructor will be used for
   * creating an instance of hte class.
   * @param values The parsed program option values
   */
  my_class(
      mack::options::values const* values)
  {
    bool is_switch_set =
      values->get_boolean("switch");
    std::string default_value_option =
      values->get("default_value_option");
    int mandatory_option =
      values->cast<int>("mandatory_option");
    my_class_type_class* default_type_option =
      values->get<my_class_type_class>("default_type_option");
    my_class_type_class* mandatory_type_option =
      values->get<my_class_type_class>("mandatory_type_option");
    OPTIONAL* optional_template =
      values->get<OPTIONAL>("optional_template");
    MANDATORY* mandatory_template =
      values->get<MANDATORY>("mandatory_template");
    ...
    if (mandatory_option < 0)
    {
      BOOST_THROW_EXCEPTION(mack::options::invalid_value_error()
          << mack::options::errinfo_option_flag("mandatory_option")
          << mack::options::errinfo_option_value(mandatory_option)
          << mack::options::errinfo_option_value_description("x >= 0"));
    }
    ...
    delete default_type_option;
    delete mandatory_type_option;
    delete optional_template;
    delete mandatory_template;
  }

  ...
}; // class my_class

} // namespace my_namespace
