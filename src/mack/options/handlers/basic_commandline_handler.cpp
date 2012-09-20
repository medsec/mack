#include "basic_commandline_handler.hpp"

#include <iostream>
#include <boost/algorithm/string/trim.hpp>
#include <mack/options/commandline_formatter.hpp>

#include <mack/options/exit_requested.hpp>
#include <cstdlib>

mack::options::handlers::basic_commandline_handler::basic_commandline_handler(
    mack::options::values const* values)
  : handler(),
    _has_been_run(false),
    _repeat(values->get_boolean("repeat")),
    _input_stream(std::cin),
    _output_stream(std::cout),
    _error_stream(std::cerr)
{
}

mack::options::handlers::basic_commandline_handler::basic_commandline_handler(
    const bool repeat,
    std::istream& input_stream,
    std::ostream& message_output_stream,
    std::ostream& error_output_stream)
  : handler(),
    _has_been_run(false),
    _repeat(repeat),
    _input_stream(input_stream),
    _output_stream(message_output_stream),
    _error_stream(error_output_stream)
{
}

mack::options::handlers::basic_commandline_handler::~basic_commandline_handler()
{
}

mack::options::values const*
mack::options::handlers::basic_commandline_handler::run()
{
  if (_has_been_run && !_repeat)
  {
    throw mack::options::exit_requested(EXIT_SUCCESS);
  }
  else
  {
    _has_been_run = true;
  }

  _output_stream << std::endl;
  _output_stream << "-=Basic Commandline Handler=-" << std::endl;
  _output_stream << "Type in ':help' for help." << std::endl;
  values const* vals = NULL;
  while (vals == NULL)
  {
    bool is_command = false;
    std::string input;
    while (!is_command)
    {
      _output_stream << "> ";
      std::getline(_input_stream, input);

      if (!input.empty())
      {
        if (input[0] == ':')
        {
          input = input.substr(1);
          boost::algorithm::trim(input);
          is_command = true;
        }
        else
        {
          boost::algorithm::trim(input);
          set_value_from_input(input);
        }
      }
    }
    vals = execute_command(input);
  }
  return vals;
}

mack::options::values const*
mack::options::handlers::basic_commandline_handler::execute_command(
    std::string const& command) const
{
  if (command == "help")
  {
    _output_stream << "Commands:" << std::endl;
    _output_stream << "  :help" << std::endl
      << "    Shows this message." << std::endl; 
    _output_stream << "  :quit" << std::endl
      << "    Exits the program." << std::endl; 
    _output_stream << "  :show" << std::endl
      << "    Shows the current settings." << std::endl; 
    _output_stream << "  :run" << std::endl
      << "    Runs the program with current settings." << std::endl; 
    _output_stream << "Settings can be changed by using the syntax:"
      << std::endl;
    _output_stream << "  <flag>" << std::endl;
    _output_stream << "or" << std::endl;
    _output_stream << "  <flag> = <value>" << std::endl;
    _output_stream << "where <flag> is the flag of an option including the option "
      << "namespace." << std::endl;
    _output_stream << "For example: 'my_namespace.my_option_flag = my_value'"
      << std::endl;
  }
  else if (command == "quit")
  {
    throw mack::options::exit_requested(EXIT_SUCCESS);
  }
  else if (command == "show")
  {
    print_options();
  }
  else if (command == "run")
  {
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
    }
  }
  else
  {
    _output_stream << "Unknown command ':" << command << "'" << std::endl;
    _output_stream << "Type in ':help' for help." << std::endl;
  }
  return NULL;
}

void
mack::options::handlers::basic_commandline_handler::set_value_from_input(
    std::string const& input)
{
  bool was_successfull;
  std::string error_message;

  const size_t equals_index = input.find('=');
  if (equals_index == std::string::npos)
  {
    // check for space
    const size_t space_index = input.find_first_of(" \t");
    if (space_index == std::string::npos)
    {
      set(input, was_successfull, error_message);
    }
    else
    {
      // get value and flag
      const std::string flag =
        boost::algorithm::trim_copy(input.substr(0, space_index));
      const std::string value =
        boost::algorithm::trim_copy(input.substr(space_index + 1));
      set(flag, value, was_successfull, error_message);
    }
  }
  else // (equals_index != std::string::npos)
  {
    // get value and flag
    const std::string flag =
      boost::algorithm::trim_copy(input.substr(0, equals_index));
    const std::string value =
      boost::algorithm::trim_copy(input.substr(equals_index + 1));
    set(flag, value, was_successfull, error_message);
  }

  if (!was_successfull)
  {
    _error_stream << error_message << std::endl;
  }
}

void
mack::options::handlers::basic_commandline_handler::print_options() const
{
  commandline_formatter formatter(_output_stream, false);

  formatter.print_description(
      get_program_name(),
      get_brief_program_description(),
      get_full_program_description());

  std::vector<option const*> options = get_options();
  for (std::vector<option const*>::const_iterator options_it = options.begin();
      options_it != options.end();
      ++options_it)
  {
    formatter.print(*options_it);
  }
}

