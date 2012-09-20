#include "option.hpp"

#include <mack/options/options.hpp>

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>

mack::options::option::option()
  : _short_flag(),
    _long_flag(),
    _brief(),
    _full(),
    _has_default_value(false),
    _default_value(),
    _has_value(false),
    _value()
{
}
 
mack::options::option::option(
    option const& original)
  : _short_flag(original._short_flag),
    _long_flag(original._long_flag),
    _brief(original._brief),
    _full(original._full),
    _has_default_value(original._has_default_value),
    _default_value(original._default_value),
    _has_value(original._has_value),
    _value(original._value)
{
}

mack::options::option::option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description)
  : _short_flag(short_flag),
    _long_flag(long_flag),
    _brief(brief_description),
    _full(create_full_description(brief_description, detailed_description)),
    _has_default_value(false),
    _default_value(),
    _has_value(false),
    _value()
{
}

mack::options::option::option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    std::string const& default_value)
  : _short_flag(short_flag),
    _long_flag(long_flag),
    _brief(brief_description),
    _full(create_full_description(brief_description, detailed_description)),
    _has_default_value(true),
    _default_value(default_value),
    _has_value(true),
    _value(default_value)
{
}

mack::options::option::option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    std::string const& default_value,
    std::string const& start_value)
  : _short_flag(short_flag),
    _long_flag(long_flag),
    _brief(brief_description),
    _full(create_full_description(brief_description, detailed_description)),
    _has_default_value(true),
    _default_value(default_value),
    _has_value(true),
    _value(start_value)
{
}

mack::options::option::~option()
{
}

mack::options::option&
mack::options::option::operator=(
    mack::options::option const& original)
{
  _short_flag = original._short_flag;
  _long_flag = original._long_flag;
  _brief = original._brief;
  _full = original._full;
  _has_default_value = original._has_default_value;
  _default_value = original._default_value;
  _has_value = original._has_value;
  _value = original._value;
  return *this;
}

std::string const&
mack::options::option::get_short_flag() const
{
  return _short_flag;
}

std::string const&
mack::options::option::get_long_flag() const
{
  return _long_flag;
}

std::string const&
mack::options::option::get_brief_description() const
{
  return _brief;
}

std::string const&
mack::options::option::get_full_description() const
{
  return _full;
}

bool
mack::options::option::has_value() const
{
  return _has_value;
}

std::string
mack::options::option::get_value() const
{
  if (!has_value())
  {
    BOOST_THROW_EXCEPTION(no_value_error()
        << errinfo_option_flag(get_long_flag()));
  }
  else
  {
    return _value;
  }
}

mack::options::value const*
mack::options::option::create() const
{
  return new value(get_value());
}

bool
mack::options::option::has_child_options() const
{
  return false;
}

mack::options::options const*
mack::options::option::get_child_options() const
{
  BOOST_THROW_EXCEPTION(no_such_namespace_error()
      << errinfo_option_flag(get_long_flag()));
}

mack::options::options*
mack::options::option::get_child_options()
{
  BOOST_THROW_EXCEPTION(no_such_namespace_error()
      << errinfo_option_flag(get_long_flag()));
}

mack::options::option const*
mack::options::option::get_child_option(
    std::string const& flag) const
{
  return get_child_options()->get_option(flag);
}

mack::options::option*
mack::options::option::get_child_option(
    std::string const& flag)
{
  return get_child_options()->get_option(flag);
}

bool
mack::options::option::is_selection() const
{
  return false;
}

std::vector<std::string>::const_iterator
mack::options::option::selection_values_begin() const
{
  BOOST_THROW_EXCEPTION(no_selection_error()
      << errinfo_option_flag(get_long_flag()));
}

std::vector<std::string>::const_iterator
mack::options::option::selection_values_end() const
{
  BOOST_THROW_EXCEPTION(no_selection_error()
      << errinfo_option_flag(get_long_flag()));
}

bool
mack::options::option::is_valid_value(
    std::string const& value) const
{
  if (is_selection())
  {
    for (std::vector<std::string>::const_iterator selection_it =
          selection_values_begin();
        selection_it != selection_values_end();
        ++selection_it)
    {
      if (value.compare(*selection_it) == 0)
      {
        return true;
      }
    }
    return false;
  }
  else
  {
    return true;
  }
}

void
mack::options::option::set_value()
{
  if (_has_default_value)
  {
    set_value(_default_value);
  }
  else
  {
    _has_value = false;
  }
}

void
mack::options::option::set_value(
    std::string const& value)
{
  if (!is_valid_value(value))
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_value(value)
        << errinfo_option_flag(get_long_flag()));
  }
  else
  {
    _value = value;
    _has_value = true;
  }
}

bool
mack::options::option::has_default_value() const
{
  return _has_default_value;
}

std::string const&
mack::options::option::get_default_value() const
{
  if (!has_default_value())
  {
    BOOST_THROW_EXCEPTION(no_value_error()
        << errinfo_option_flag(get_long_flag()));
  }
  else
  {
    return _default_value;
  }
}

void
mack::options::option::set_default_value(
    std::string const& default_value)
{
  if (!is_valid_value(default_value))
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_value(default_value)
        << errinfo_option_flag(get_long_flag()));
  }
  else
  {
    _default_value = default_value;
  }
}

std::string
mack::options::option::create_full_description(
    std::string const& brief_description,
    std::string const& detailed_description)
{
  if (detailed_description.empty())
  {
    return brief_description;
  }
  else
  {
    return brief_description + "\n" + detailed_description;
  }
}

