#include "value.hpp"

#include <mack/options/exceptions.hpp>

mack::options::value::value()
  : _is_class_type(false),
    _value(),
    _value_class_name()
{
}

mack::options::value::value(
    std::string const& value)
  : _is_class_type(false),
    _value(value),
    _value_class_name()
{
}

mack::options::value::value(
    const char* value)
  : _is_class_type(false),
    _value(std::string(value)),
    _value_class_name()
{
}

mack::options::value::value(
    mack::options::value const& original)
  : _is_class_type(original._is_class_type),
    _value(original._value),
    _value_class_name(original._value_class_name)
{
}

mack::options::value::~value()
{
}

std::string
mack::options::value::get() const
{
  return boost::any_cast<std::string>(_value);
}

bool
mack::options::value::get_boolean() const
{
  try
  {
    const std::string value = get();
    if (value.size() == 1 && value[0] == '1')
    {
      return true;
    }
    else if (value.size() == 1 && value[0] == '0')
    {
      return false;
    }
    else if (value.size() == 4
        && (value[0] == 't' || value[0] == 'T')
        && (value[1] == 'r' || value[1] == 'R')
        && (value[2] == 'u' || value[2] == 'U')
        && (value[3] == 'e' || value[3] == 'E'))
    {
      return true;
    }
    else if (value.size() == 5
        && (value[0] == 'f' || value[0] == 'F')
        && (value[1] == 'a' || value[1] == 'A')
        && (value[2] == 'l' || value[2] == 'L')
        && (value[3] == 's' || value[3] == 'S')
        && (value[4] == 'e' || value[4] == 'E'))
    {
      return false;
    }
    else
    {
      BOOST_THROW_EXCEPTION(invalid_type_error()
          << errinfo_option_value(value)
          << errinfo_option_type("boolean"));
    }
  }
  catch (boost::bad_any_cast e)
  {
      BOOST_THROW_EXCEPTION(invalid_type_error()
          << errinfo_option_type("boolean"));
  }
}


std::string const&
mack::options::value::get_value_class_name() const
{
  if (!_is_class_type)
  {
    BOOST_THROW_EXCEPTION(invalid_type_error());
  }
  else
  {
    return _value_class_name;
  }
}

