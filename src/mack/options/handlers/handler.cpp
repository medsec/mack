#include "handler.hpp"

#include <mack/options/exceptions.hpp>

#include <mack/core/null_pointer_error.hpp>

mack::options::handlers::handler::handler()
  : _options(NULL)
{
}

mack::options::handlers::handler::~handler()
{
}

void
mack::options::handlers::handler::set_program_options(
    mack::options::program_options* options)
{
  if (options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  _options = options;
}

std::string const&
mack::options::handlers::handler::get_program_name() const
{
  return options()->get_program_name();
}

std::string const&
mack::options::handlers::handler::get_brief_program_description() const
{
  return options()->get_brief_description();
}

std::string const&
mack::options::handlers::handler::get_full_program_description() const
{
  return options()->get_full_description();
}

std::vector<mack::options::option const*>
mack::options::handlers::handler::get_options() const
{
  std::vector<option const*> all_options;

  all_options.insert(
      all_options.end(), options()->begin(), options()->end());
  program_options const* const_options = options();
  std::vector<option const*> default_options =
    const_options->get_default_options();
  all_options.insert(
      all_options.end(), default_options.begin(), default_options.end());

  return all_options;
}

void
mack::options::handlers::handler::set(
    std::string const& flag,
    bool& was_successfull,
    std::string& error_message)
{
  was_successfull = false;
  CATCH_AND_GET_MESSAGE(
      options()->set(flag);
      was_successfull = true;,
      error_message);
}

void
mack::options::handlers::handler::set(
    std::string const& flag,
    std::string const& value,
    bool& was_successfull,
    std::string& error_message)
{
  was_successfull = false;
  CATCH_AND_GET_MESSAGE(
    options()->set(flag, value);
    was_successfull = true;,
    error_message);
}

mack::options::values const*
mack::options::handlers::handler::create(
    bool& was_successfull,
    std::string& error_message) const
{
  was_successfull = false;
  CATCH_AND_GET_MESSAGE(
    values const* vals = options()->create();
    was_successfull = true;
    return vals;,
    error_message);
  return NULL;
}

mack::options::program_options*
mack::options::handlers::handler::options()
{
  if (_options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  return _options;
}

mack::options::program_options const*
mack::options::handlers::handler::options() const
{
  if (_options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  return _options;
}

