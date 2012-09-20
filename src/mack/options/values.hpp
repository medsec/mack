#ifndef __MACK_OPTIONS_VALUES_HPP__
#define __MACK_OPTIONS_VALUES_HPP__

#include <mack/options/value.hpp>
#include <mack/options/exceptions.hpp>

#include <string>
#include <map>
#include <typeinfo>

namespace mack {
namespace options {

/**
 * @brief A map of values created from a set of options.
 * @details This container allows to access the values obtained on parsing
 * the program options by using the <i>long</i> flags of the respective
 * options (without the option namespace).
 * These values can either be strings or pointers to an object.
 *
 * If a value is a pointer to an object, the object will not be deleted on the
 * deletion of this object. A reference to the object can be attained by
 * using the template <tt>get</tt> method.
 *
 * If the value is a string, it can be accessed by using the normal
 * <tt>get</tt> method. If the string should be of a certain primitive type
 * (e.g., <tt>int</tt>, <tt>float</tt>, ...), the <tt>cast</tt> method can be
 * employed in order to cast the string to the desired type.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see mack::options
 * @see mack::options::option
 * @see mack::options::options
 * @see mack::options::value
 */
class values {
  public:

    /**
     * @brief Creates a new and empty values object.
     */
    values();

    /**
     * @brief Creates a new values object from the provided map.
     * @param values a map from <i>long</i> flags to the corresponding value of
     * an option.
     * @throws null_pointer_error if one of the value pointers is <tt>null</tt>
     */
    values(
        std::map<std::string, value const*> const& values);

    /**
     * @brief The destructor.
     * @details All <tt>value</tt> objects inside of the map are deleted.
     */
    ~values();

    /**
     * @brief Checks if this values object contains no values.
     * @return if no values are specified for this object
     */
    bool
    is_empty() const;

    /**
     * @brief Gets the string value assigned to a certain option.
     * @param long_flag the flag of the option
     * @return the assigned value
     * @throws no_such_option_error if no option with given flag exists
     * @throws invalid_type_error if the assigned value is not a string,
     * but an object of some option type
     */
    std::string
    get(
        std::string const& long_flag) const;

    /**
     * @brief Gets the boolean value assigned to a certain option switch.
     * @param long_flag the flag of the option switch
     * @return if the switch was set
     * @throws no_such_option_error if no option with given flag exists
     * @throws invalid_type_error if the assigned value is not boolean
     */
    bool
    get_boolean(
        std::string const& long_flag) const;

    /**
     * @brief Gets the value assigned to a certain type option.
     * @tparam T the type of the assigned value
     * @param long_flag the flag of the type option
     * @return a pointer to the created type object
     * @throws no_such_option_error if no option with given flag exists
     * @throws invalid_type_error if the assigned value is not of type T
     */
    template<class T>
    T*
    get(
        std::string const& long_flag) const
    {
      try
      {
        return find(long_flag)->get<T>();
      }
      catch (boost::bad_any_cast e)
      {
        BOOST_THROW_EXCEPTION(invalid_type_error()
            << errinfo_option_type(typeid(T).name())
            << errinfo_option_flag(long_flag));
      }
    }

    /**
     * @brief Gets the string value assigned to a certain option and casts it to
     * a different type.
     * @details This method employs <tt>boost::lexical_cast</tt> for casting
     * strings to different types.
     * @tparam T the type to cast the string value to
     * @param long_flag the flag of the type option
     * @return a pointer the casted value
     * @throws no_such_option_error if no option with given flag exists
     * @throws invalid_type_error if the assigned value can not be cast to T
     */
    template<class T>
    T
    cast(
        std::string const& long_flag) const
    {
      try
      {
        return find(long_flag)->cast<T>();
      }
      catch (boost::bad_any_cast e)
      {
        BOOST_THROW_EXCEPTION(invalid_type_error()
            << errinfo_option_type(typeid(T).name())
            << errinfo_option_flag(long_flag));
      }
      catch (boost::bad_lexical_cast e)
      {
        BOOST_THROW_EXCEPTION(invalid_type_error()
            << errinfo_option_value(get(long_flag))
            << errinfo_option_type(typeid(T).name())
            << errinfo_option_flag(long_flag));
      }
    }

    /**
     * @brief Gets the class name of the value with given flag.
     * @details If the value contains directly a string, an error
     * is thrown.
     * @param long_flag the flag of the type option
     * @return the full class name of the class of the value with given flag
     * @throws no_such_option_error if no option with given flag exists
     * @throws invalid_type_error if the value is a string
     */
    std::string const&
    get_value_class_name(
        std::string const& long_flag) const;

  private:

    /**
     * @brief The copy constructor.
     * @param original the values to copy from
     */
    values(
        values const& original);

    /**
     * @brief The assignment operator.
     * @param original the values to copy from
     * @return this object
     */
    values&
    operator=(
        values const& original);

    /**
     * @brief Gets the value with given flag.
     * @details If no such value exists, a no_such_option_error is thrown.
     * @param long_flag the flag of the value to be returned
     * @return the value
     * @throws no_such_option_error if there is no value with given flag
     */
    value const*
    find(
        std::string const& long_flag) const;

    /**
     * The values as a mapping from <i>long</i> flag to value pointer.
     */
    std::map<std::string, value const*> _values;

}; // class values

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_VALUES_HPP__ */
