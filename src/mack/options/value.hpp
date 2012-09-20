#ifndef __MACK_OPTIONS_VALUE_HPP__
#define __MACK_OPTIONS_VALUE_HPP__

#include <string>
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>

namespace mack {
namespace options {

/**
 * @brief A single value created from an option.
 * @details This container can hold either a string or a pointer to an
 * object. Which type of object is hold depends on the used constructor.
 *
 * If a pointer to an object is hold, the object will <b>not</b> be deleted on
 * the deletion of this object. A reference to the object can be attained by
 * using the template <tt>get</tt> method.
 *
 * If the object is a string, it can be accessed by using the normal
 * <tt>get</tt> method. If the string should be of a certain primitive type
 * (e.g., <tt>int</tt>, <tt>float</tt>, ...), the <tt>cast</tt> method can be
 * employed in order to cast the string to the desired type.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 * @see mack::options
 * @see mack::options::option
 */
class value {

  public:

    /**
     * @brief Creates a new value which holds a string.
     *
     * @param value the string to be hold
     */
    value(
        std::string const& value);

    /**
     * @brief Creates a new value which holds a string.
     *
     * @param value the string to be hold
     */
    value(
        const char* value);

    /**
     * @brief Creates a new value that holds a reference to an object of type
     * <tt>T</tt>.
     * @details The given object will <b>not</b> be deleted if the destructor
     * of this object is called.
     *
     * @tparam T the type of the object
     * @param value a pointer to the object to be hold. Note that if the pointer
     * is <tt>const</tt>, the <tt>const</tt> keyword will also have to be
     * applied in the template parameter of the <tt>get</tt> method
     * @param class_name the full qualified name of T
     */
    template<class T>
    value(
        T* value,
        std::string const& class_name)
      : _is_class_type(true),
        _value(value),
        _value_class_name(class_name)
    {
    }

    /**
     * @brief The copy constructor.
     * @param original the value to copy from.
     */
    value(
        value const& original);

    /**
     * @brief The destructor.
     */
    ~value();

    /**
     * @brief Gets the string hold by this object.
     *
     * @throws boost::bad_any_cast if this object does not contain a string but
     * an object of different type
     */
    std::string
    get() const;

    /**
     * @brief Gets the boolean value assigned to this option.
     * @return if this switch was set
     * @throws invalid_type_error if the assigned value is not boolean
     */
    bool
    get_boolean() const;

    /**
     * @brief Gets the object hold by this <tt>value</tt> object.
     *
     * @tparam T the type of the hold object (not the pointer type)
     * @throws boost::bad_any_cast if the hold object is not of the type given
     * as template parameter
     */
    template<class T>
    T*
    get() const
    {
      return boost::any_cast<T*>(_value);
    }

    /**
     * @brief Casts the string hold by this object to a certain type and
     * returns it.
     * @details This method employs <tt>boost::lexical_cast</tt> for casting
     * the string to the certain type.
     *
     * @tparam T the type to cast the string to
     * @throws boost::bad_any_cast if this object does not contain a string but
     * an object of different type
     * @throws boost::bad_lexical_cast if the string could not be cast to
     * given type
     */
    template<class T>
    T
    cast() const
    {
      return boost::lexical_cast<T>(get());
    }

    /**
     * @brief Gets the class name of the type contained in this object.
     * @details If this object contains directly a string value, an error
     * is thrown.
     * @return the full class name of the object class
     * @throws invalid_type_error if this value is a string
     */
    std::string const&
    get_value_class_name() const;

  private:

    /**
     * @brief The default constructor.
     */
    value();

    /**
     * If the value contains a pointer to an instance of an option type.
     */
    bool _is_class_type;

    /**
     * The value contained in the <tt>boost::any</tt> container.
     */
    boost::any _value;

    /**
     * The name of the class of the option.
     */
    std::string _value_class_name;

}; // class value

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_VALUE_HPP__ */

