#include "parser.hpp"

#include <mack/options/programs.hpp>

#include <mack/options/exceptions.hpp>

#include <mack/core/null_pointer_error.hpp>

#include <mack/options/exit_requested.hpp>
#include <cstdlib>

mack::options::parser::parser()
  : _values(NULL),
    _program_options(NULL),
    _interaction_handler(NULL),
    _message_stream(std::cout),
    _error_stream(std::cerr)
{
}

mack::options::parser::parser(
    const int argc,
    char const* const* argv,
    mack::options::program_options* program_options,
    std::ostream& message_output_stream,
    std::ostream& error_output_stream)
  : _values(NULL),
    _program_options(program_options),
    _interaction_handler(NULL),
    _message_stream(message_output_stream),
    _error_stream(error_output_stream)
{
  if (_program_options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  set_program_options(argc, argv);
  check_help();
  set_interaction_handler();
}

mack::options::parser::parser(
    const int argc,
    char const* const* argv,
    std::string const& program_name,
    std::ostream& message_output_stream,
    std::ostream& error_output_stream)
  : _values(NULL),
    _program_options(NULL),
    _interaction_handler(NULL),
    _message_stream(message_output_stream),
    _error_stream(error_output_stream)
{
  set_program(program_name);
  set_program_options(argc, argv);
  check_help();
  set_interaction_handler();
}

mack::options::parser::~parser()
{
  if (_values != NULL)
  {
    delete _values;
  }
  if (_program_options != NULL)
  {
    delete _program_options;
  }
  if (_interaction_handler != NULL)
  {
    delete _interaction_handler;
  }
}

mack::options::values const*
mack::options::parser::parse()
{
  if (_values != NULL)
  {
    delete _values; // values of last parsing
    _values = NULL;
  }

  _values = _interaction_handler->run();
  return _values;
}

void
mack::options::parser::check_help()
{
  if (_program_options->is_help_set())
  {
    _program_options->print_help(_message_stream);
    throw exit_requested(EXIT_SUCCESS);
  }
}

void
mack::options::parser::set_program(
    std::string const& program_name)
{
  bool was_successfull = false;
  std::string error_message;

  // Getting program options
  try
  {
    CATCH_AND_GET_MESSAGE(
      _program_options =
        mack::options::programs::create_program_option(program_name);
      was_successfull = true,
      error_message);
  }
  catch (mack::options::programs::no_such_program_error e)
  {
    _error_stream << "Unknown program name: " << program_name << std::endl;
    _error_stream << "Please confirm that the program name provided in the "
      << "annotation of the program (@program <program name>) is identical to "
      << "the name given to the constructor of the mack::options::parser "
      << "object in the main body of the program." << std::endl;
    _error_stream << "If so, it may be necessary to rerun doxygen:" << std::endl;
    _error_stream << "> make update" << std::endl;
    throw exit_requested(EXIT_FAILURE);
  }

  if (!was_successfull)
  {
    _error_stream << "Error while creating program options:" << std::endl;
    _error_stream << error_message << std::endl;
    throw exit_requested(EXIT_FAILURE);
  }
}

void
mack::options::parser::set_program_options(
    const int argc,
    char const* const* argv)
{
  bool was_successfull = false;
  std::string error_message;

  // Setting program options
  CATCH_AND_GET_MESSAGE(
    _program_options->set_all(argc, argv);
    _program_options->set_commandline_arguments_done();
    was_successfull = true;,
    error_message);

  if (!was_successfull)
  {
    _error_stream << "Error while setting program options:" << std::endl;
    _error_stream << error_message << std::endl;
    throw exit_requested(EXIT_FAILURE);
  }
}

void
mack::options::parser::set_interaction_handler()
{
  bool was_successfull = false;
  std::string error_message;

  // Creating interaction handler
  CATCH_AND_GET_MESSAGE(
    _interaction_handler = _program_options->create_interaction_handler();
    _interaction_handler->set_program_options(_program_options);
    was_successfull = true;,
    error_message);

  if (!was_successfull)
  {
    _error_stream << "Error while creating interaction handler:" << std::endl;
    _error_stream << error_message << std::endl;
    throw exit_requested(EXIT_FAILURE);
  }
}

