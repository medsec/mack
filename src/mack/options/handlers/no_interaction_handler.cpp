#include "no_interaction_handler.hpp"

#include <iostream>

#include <mack/options/exit_requested.hpp>
#include <cstdlib>

mack::options::handlers::no_interaction_handler::no_interaction_handler(
        std::ostream& error_output_stream)
  : _has_been_run(false),
    _error_stream(error_output_stream)
{
}

mack::options::handlers::no_interaction_handler::~no_interaction_handler()
{
}

mack::options::values const*
mack::options::handlers::no_interaction_handler::run()
{
  if (_has_been_run)
  {
    throw mack::options::exit_requested(EXIT_SUCCESS);
  }
  else
  {
    _has_been_run = true;
  }

  bool was_successfull;
  std::string error_message;

  values const* vals = create(was_successfull, error_message);

  if (was_successfull)
  {
    return vals;
  }
  else
  {
    _error_stream << error_message << std::endl;
    throw mack::options::exit_requested(EXIT_FAILURE);
  }
}

