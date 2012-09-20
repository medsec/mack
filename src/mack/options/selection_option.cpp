#include "selection_option.hpp"

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>

mack::options::selection_option::selection_option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    std::vector<std::string> const& selection_values)
  : mack::options::option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description),
   _selection_values(selection_values)
{
  if (_selection_values.empty())
  {
    BOOST_THROW_EXCEPTION(no_selection_error()
        << errinfo_option_flag(long_flag));
  }
}

mack::options::selection_option::selection_option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    std::vector<std::string> const& selection_values,
    std::string const& default_value)
  : mack::options::option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description,
      default_value),
    _selection_values(selection_values)
{
  if (_selection_values.empty())
  {
    BOOST_THROW_EXCEPTION(no_selection_error()
        << errinfo_option_flag(long_flag));
  }
  if (!is_valid_value(default_value))
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_flag(get_long_flag())
        << errinfo_option_value(default_value));
  }
}

mack::options::selection_option::selection_option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    std::vector<std::string> const& selection_values,
    std::string const& default_value,
    std::string const& start_value)
  : mack::options::option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description,
      default_value,
      start_value),
    _selection_values(selection_values)
{
  if (_selection_values.empty())
  {
    BOOST_THROW_EXCEPTION(no_selection_error()
        << errinfo_option_flag(long_flag));
  }
  if (!is_valid_value(default_value))
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_flag(get_long_flag())
        << errinfo_option_value(default_value));
  }
  else if (!is_valid_value(start_value))
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_flag(get_long_flag())
        << errinfo_option_value(start_value));
  }
}

mack::options::selection_option::~selection_option()
{
}

bool
mack::options::selection_option::is_selection() const
{
  return true;
}

std::vector<std::string>::const_iterator
mack::options::selection_option::selection_values_begin() const
{
  return _selection_values.begin();
}

std::vector<std::string>::const_iterator
mack::options::selection_option::selection_values_end() const
{
  return _selection_values.end();
}

