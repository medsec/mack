#include "values.hpp"

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>
#include <mack/core/null_pointer_error.hpp>

mack::options::values::values()
  : _values()
{
}

mack::options::values::values(
    mack::options::values const& original)
  : _values(original._values)
{
}

mack::options::values::values(
    std::map<std::string, value const*> const& values)
  : _values(values)
{
  for (std::map<std::string, value const*>::iterator values_it = _values.begin();
      values_it != _values.end();
      ++values_it)
  {
    if (values_it->second == NULL)
    {
      BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
    }
  }
}

mack::options::values::~values()
{
  for (std::map<std::string, value const*>::iterator values_it = _values.begin();
      values_it != _values.end();
      ++values_it)
  {
    delete values_it->second;
  }
  _values.clear();
}

mack::options::values&
mack::options::values::operator=(
    mack::options::values const& original)
{
  _values = original._values;
  return *this;
}

bool
mack::options::values::is_empty() const
{
  return _values.empty();
}

std::string
mack::options::values::get(
    std::string const& long_flag) const
{
  try
  {
    return find(long_flag)->get();
  }
  catch (boost::bad_any_cast e)
  {
    BOOST_THROW_EXCEPTION(invalid_type_error()
        << errinfo_option_type("string")
        << errinfo_option_flag(long_flag));
  }
}

bool
mack::options::values::get_boolean(
    std::string const& long_flag) const
{
  try
  {
    return find(long_flag)->get_boolean();
  }
  catch (invalid_type_error e)
  {
    e << errinfo_option_flag(long_flag);
    throw;
  }
}

std::string const&
mack::options::values::get_value_class_name(
    std::string const& long_flag) const
{
  try
  {
    return find(long_flag)->get_value_class_name();
  }
  catch (invalid_type_error e)
  {
    e << errinfo_option_flag(long_flag);
    throw;
  }
}

mack::options::value const*
mack::options::values::find(
    std::string const& long_flag) const
{
  std::map<std::string, value const*>::const_iterator value_position =
    _values.find(long_flag);

  if (value_position != _values.end())
  {
    return value_position->second;
  }
  else
  {
    BOOST_THROW_EXCEPTION(no_such_option_error()
        << errinfo_option_flag(long_flag));
  }
}

