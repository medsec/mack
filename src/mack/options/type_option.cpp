#include "type_option.hpp"

#include <mack/options/values.hpp>

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <boost/algorithm/string/predicate.hpp>

mack::options::type_option::type_option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    const bool is_for_template,
    mack::options::option_type const* type)
  : mack::options::option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description),
    _type(type),
    _options(NULL),
    _is_for_template(is_for_template)
{
  if (_type == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
}

mack::options::type_option::type_option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    const bool is_for_template,
    mack::options::option_type const* type,
    std::string const& default_value)
  : mack::options::option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description,
      default_value),
    _type(type),
    _options(NULL),
    _is_for_template(is_for_template)
{
  if (_type == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  set_default_value(get_class_name(default_value)); // throws if invalid
  set_value();
}

mack::options::type_option::type_option(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description,
    const bool is_for_template,
    mack::options::option_type const* type,
    std::string const& default_value,
    std::string const& start_value)
  : mack::options::option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description,
      default_value),
    _type(type),
    _options(NULL),
    _is_for_template(is_for_template)
{
  if (_type == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  set_default_value(get_class_name(default_value)); // throws if invalid
  set_value(start_value); // throws if invalid
}

mack::options::type_option::~type_option()
{
  delete _type;
  delete _options;
}

mack::options::value const*
mack::options::type_option::create() const
{
  std::string const& class_name = get_value(); // throws if not set
  values const* vals = _options->create();
  value const* val = _type->create_value(
      class_name, vals, _is_for_template);
  delete vals;
  return val;
}

bool
mack::options::type_option::has_child_options() const
{
  return has_value();
}

mack::options::options const*
mack::options::type_option::get_child_options() const
{
  if (!has_child_options())
  {
    option::get_child_options(); // throws error
  }
  return _options;
}

mack::options::options*
mack::options::type_option::get_child_options()
{
  if (!has_child_options())
  {
    option::get_child_options(); // throws error
  }
  return _options;
}

mack::options::option const*
mack::options::type_option::get_child_option(
    std::string const& flag) const
{
  return get_child_options()->get_option(flag);
}

mack::options::option*
mack::options::type_option::get_child_option(
    std::string const& flag)
{
  return get_child_options()->get_option(flag);
}

bool
mack::options::type_option::is_selection() const
{
  return true;
}

std::vector<std::string>::const_iterator
mack::options::type_option::selection_values_begin() const
{
  return _type->get_class_names().begin();
}

std::vector<std::string>::const_iterator
mack::options::type_option::selection_values_end() const
{
  return _type->get_class_names().end();
}

bool
mack::options::type_option::is_valid_value(
    std::string const& value) const
{
  try
  {
    get_class_name(value);
    return true;
  }
  catch (ambiguous_value_error e)
  {
    return false;
  }
  catch (invalid_value_error e)
  {
    return false;
  }
}

void
mack::options::type_option::set_value()
{
  if (has_default_value())
  {
    set_value(get_default_value());
  }
  else
  {
    clear_options();
    option::set_value();
  }
}

void
mack::options::type_option::set_value(
    std::string const& value)
{
  clear_options();
  const std::string class_name = get_class_name(value); // throws if invalid
  option::set_value(class_name);
  _options = _type->get_options(class_name);
}

void
mack::options::type_option::clear_options()
{
  if (_options != NULL)
  {
    delete _options;
    _options = NULL;
  }
}

std::string
mack::options::type_option::get_class_name(
    std::string const& value) const
{
  // Check for exact match
  for (std::vector<std::string>::const_iterator selections_it =
        selection_values_begin();
      selections_it != selection_values_end();
      ++selections_it)
  {
    if (value.compare(*selections_it) == 0)
    {
      return *selections_it;
    }
  }

  bool found = false;
  std::string found_name = "";
  // Check for case insensitive match
  for (std::vector<std::string>::const_iterator selections_it =
        selection_values_begin();
      selections_it != selection_values_end();
      ++selections_it)
  {
    if (boost::iequals(value, *selections_it))
    {
      if (found)
      { // ambiguous
        BOOST_THROW_EXCEPTION(ambiguous_value_error()
            << errinfo_option_value(value)
            << errinfo_option_value_senses(
              std::pair<std::string, std::string>(found_name, *selections_it))
            << errinfo_option_flag(get_long_flag()));
      }
      else
      {
        found = true;
        found_name = *selections_it;
      }
    }
  }
  if (found)
  {
    return found_name;
  }


  // Check for missing namespace
  if (value.find("::") == std::string::npos)
  {
    // Check case sensitive
    for (std::vector<std::string>::const_iterator selections_it =
          selection_values_begin();
        selections_it != selection_values_end();
        ++selections_it)
    {
      const size_t namespace_separator_index = selections_it->rfind("::");
      if (namespace_separator_index == std::string::npos
          || namespace_separator_index == selections_it->length() - 2)
      {
        continue; // has no namespace or '::' is the last symbol
      }
      const std::string class_name =
        selections_it->substr(namespace_separator_index + 2);

      if (value.compare(class_name) == 0)
      {
        if (found)
        { // ambiguous
          BOOST_THROW_EXCEPTION(ambiguous_value_error()
              << errinfo_option_value(value)
              << errinfo_option_value_senses(
                std::pair<std::string, std::string>(found_name, *selections_it))
              << errinfo_option_flag(get_long_flag()));
        }
        else
        {
          found = true;
          found_name = *selections_it;
        }
      }
    }
    if (found)
    {
      return found_name;
    }

    // Check case insensitive
    for (std::vector<std::string>::const_iterator selections_it =
          selection_values_begin();
        selections_it != selection_values_end();
        ++selections_it)
    {
      const size_t namespace_separator_index = selections_it->rfind("::");
      if (namespace_separator_index == std::string::npos
          || namespace_separator_index == selections_it->length() - 2)
      {
        continue; // has no namespace or '::' is the last symbol
      }
      const std::string class_name =
        selections_it->substr(namespace_separator_index + 2);

      if (boost::iequals(value, class_name))
      {
        if (found)
        { // ambiguous
          BOOST_THROW_EXCEPTION(ambiguous_value_error()
              << errinfo_option_value(value)
              << errinfo_option_value_senses(
                std::pair<std::string, std::string>(found_name, *selections_it))
              << errinfo_option_flag(get_long_flag()));
        }
        else
        {
          found = true;
          found_name = *selections_it;
        }
      }
    }
    if (found)
    {
      return found_name;
    }
  }

  // None found
  BOOST_THROW_EXCEPTION(invalid_value_error()
      << errinfo_option_value(value)
      << errinfo_option_flag(get_long_flag()));
}

