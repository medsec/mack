#include "options.hpp"

#include <mack/options/value.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/null_pointer_error.hpp>
#include <boost/throw_exception.hpp>

mack::options::options::options()
  : _brief(),
    _full(),
    _options(),
    _const_options()
{
}

mack::options::options::options(
    mack::options::options const& original)
  : _brief(original._brief),
    _full(original._full),
    _options(original._options),
    _const_options(original._const_options)
{
}

mack::options::options::options(
    std::string const& brief_description,
    std::string const& detailed_description,
    std::vector<option*> const& options)
  : _brief(brief_description),
    _full(brief_description),
    _options(options),
    _const_options()
{
  std::set<std::string> flags;
  for (std::vector<option*>::iterator options_it = _options.begin();
      options_it != _options.end();
      ++options_it)
  {
    // Check for null pointer
    if (*options_it == NULL)
    {
      BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
    }

    // Copy pointer for const iteration
    _const_options.push_back(*options_it);

    // Check for flag collision
    if ((*options_it)->get_short_flag() == (*options_it)->get_long_flag())
    {
      BOOST_THROW_EXCEPTION(flag_collision_error()
          << errinfo_option_flag((*options_it)->get_short_flag()));
    }
    if (flags.count((*options_it)->get_short_flag()) > 0)
    {
      BOOST_THROW_EXCEPTION(flag_collision_error()
          << errinfo_option_flag((*options_it)->get_short_flag()));
    }
    if (flags.count((*options_it)->get_long_flag()) > 0)
    {
      BOOST_THROW_EXCEPTION(flag_collision_error()
          << errinfo_option_flag((*options_it)->get_long_flag()));
    }

    // Insert for future checks
    flags.insert((*options_it)->get_short_flag());
    flags.insert((*options_it)->get_long_flag());
  }

  // Set full description as combination of brief and detailed
  if (!detailed_description.empty())
  {
    _full = brief_description + "\n" + detailed_description;
  }
}

mack::options::options::~options()
{
  for (std::vector<option*>::iterator options_it = _options.begin();
      options_it != _options.end();
      ++options_it)
  {
    delete *options_it;
  }
  _options.clear();
  _const_options.clear();
}

mack::options::options&
mack::options::options::operator=(
    mack::options::options const& original)
{
  _brief = original._brief;
  _full = original._full;
  _options = original._options;
  _const_options = original._const_options;
  return *this;
}

std::string const&
mack::options::options::get_brief_description() const
{
  return _brief;
}

std::string const&
mack::options::options::get_full_description() const
{
  return _full;
}

std::vector<mack::options::option*>::iterator
mack::options::options::begin()
{
  return _options.begin();
}

std::vector<mack::options::option const*>::const_iterator
mack::options::options::begin() const
{
  return _const_options.begin();
}

std::vector<mack::options::option*>::iterator
mack::options::options::end()
{
  return _options.end();
}

std::vector<mack::options::option const*>::const_iterator
mack::options::options::end() const
{
  return _const_options.end();
}

mack::options::option const*
mack::options::options::get_option(
    std::string const& flag) const
{
  for (std::vector<option*>::const_iterator options_it = _options.begin();
      options_it != _options.end();
      ++options_it)
  {
    if (flag.compare((*options_it)->get_long_flag()) == 0)
    {
      return *options_it;
    }
    else if (flag.compare((*options_it)->get_short_flag()) == 0)
    {
      return *options_it;
    }
  }
  BOOST_THROW_EXCEPTION(no_such_option_error()
      << errinfo_option_flag(flag));
}

mack::options::option*
mack::options::options::get_option(
    std::string const& flag)
{
  for (std::vector<option*>::iterator options_it = _options.begin();
      options_it != _options.end();
      ++options_it)
  {
    if (flag.compare((*options_it)->get_long_flag()) == 0)
    {
      return *options_it;
    }
    else if (flag.compare((*options_it)->get_short_flag()) == 0)
    {
      return *options_it;
    }
  }
  BOOST_THROW_EXCEPTION(no_such_option_error()
      << errinfo_option_flag(flag));
}

bool
mack::options::options::is_empty() const
{
  return _options.empty();
}

mack::options::values const*
mack::options::options::create() const
{
  std::map<std::string, value const*> mapping;
  for (std::vector<option const*>::const_iterator options_it = begin();
      options_it != end();
      ++options_it)
  {
    try
    {
      mapping.insert(std::pair<std::string, value const*>(
          (*options_it)->get_long_flag(),
          (*options_it)->create()));
    }
    catch (std::exception e)
    {
      // delete already created values
      for (std::map<std::string, value const*>::iterator mapping_it =
            mapping.begin();
          mapping_it != mapping.end();
          ++mapping_it)
      {
        delete mapping_it->second;
      }
      throw;
    }
  }
  return new values(mapping);
}

std::set<std::pair<std::string, mack::options::option*> >
mack::options::options::search_for_option(
    std::string const& flag,
    std::string const& this_namespace)
{
  std::set<std::pair<std::string, option*> > found;
  for (std::vector<option*>::iterator options_it = begin();
      options_it != end();
      ++options_it)
  {
    // check for option
    if (flag == (*options_it)->get_short_flag()
        || flag == (*options_it)->get_long_flag())
    {
      found.insert(std::pair<std::string, option*>(this_namespace, *options_it));
    }

    // check for child options -> recursive
    if ((*options_it)->has_child_options())
    {
      if (this_namespace.empty())
      {
        std::set<std::pair<std::string, option*> > found_recursive =
          (*options_it)->get_child_options()->search_for_option(
              flag, (*options_it)->get_long_flag());
        found.insert(found_recursive.begin(), found_recursive.end());
      }
      else
      {
        std::set<std::pair<std::string, option*> > found_recursive =
          (*options_it)->get_child_options()->search_for_option(
              flag,
              this_namespace + option_namespace_separator
                + (*options_it)->get_long_flag());
        found.insert(found_recursive.begin(), found_recursive.end());
      }
    }
  }
  return found;
}

