#include "exceptions.hpp"

#include <sstream>

#include <boost/exception/get_error_info.hpp>

#include <mack/options/options.hpp>

std::string
mack::options::get_error_message(
    mack::options::flag_collision_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);

  std::stringstream ss;

  ss << "A flag is used multiple times within ";

  if (option_namespace != NULL)
  {
    ss << "the option namespace \"" << *option_namespace << "\"";
  }
  else
  {
    ss << "an option namespace";
  }

  if (option_flag != NULL)
  {
    ss << ": " << *option_flag;
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::invalid_flag_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);

  std::stringstream ss;

  ss << "Detected an invalid flag";

  if (option_namespace != NULL)
  {
    ss << "in the option namespace \"" << *option_namespace << "\"";
  }

  if (option_flag != NULL)
  {
    ss << ": " << *option_flag;
  }
  else if (option_flag != NULL)
  {
    ss << '.';
  }

  ss << std::endl;
  ss << "Flags can contain letters and numbers at any position "
    << "and dashes and underscores at any but the first position.";

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::invalid_type_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);
  std::string const* option_value =
    boost::get_error_info<errinfo_option_value>(error);
  std::string const* option_type =
    boost::get_error_info<errinfo_option_type>(error);

  std::stringstream ss;

  ss << "Invalid type for option";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << " \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " in option namespace \"" << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << " \"" << *option_flag << "\"";
  }

  if (option_value != NULL)
  {
    ss << " with value \"" << *option_value << "\"";
  }

  if (option_type != NULL)
  {
    ss << " (requested type was \"" << *option_type << "\")";
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::invalid_value_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);
  std::string const* option_value =
    boost::get_error_info<errinfo_option_value>(error);
  std::string const* option_value_description =
    boost::get_error_info<errinfo_option_value_description>(error);

  std::stringstream ss;

  ss << "Invalid value for option";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << " \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " in option namespace \"" << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << " \"" << *option_flag << "\"";
  }

  if (option_value != NULL)
  {
    ss << ": \"" << *option_value << "\"";
  }

  if (option_value_description != NULL)
  {
    ss << " (constraint: \"" << *option_value_description << "\")";
  }

  size_t const* configuration_file_line =
    boost::get_error_info<mack::core::files::errinfo_parse_line>(error);

  std::string const* configuration_file =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  if (configuration_file_line != NULL || configuration_file != NULL)
  {
    ss << " from configuration file";
    if (configuration_file != NULL)
    {
      ss << " \"" << *configuration_file << "\"";
    }
    if (configuration_file_line != NULL)
    {
      ss << " in line " << configuration_file_line;
    }
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::ambiguous_value_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);
  std::string const* option_value =
    boost::get_error_info<errinfo_option_value>(error);
  std::pair<std::string, std::string> const* option_value_senses =
    boost::get_error_info<errinfo_option_value_senses>(error);

  std::stringstream ss;

  ss << "Ambiguous value";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << " for option \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " in option namespace \"" << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << " for option \"" << *option_flag << "\"";
  }

  if (option_value != NULL)
  {
    ss << ": \"" << *option_value << "\"";
  }

  if (option_value_senses != NULL)
  {
    ss << " (could be \"" << option_value_senses->first << "\" or \""
      << option_value_senses->second << "\")";
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::unbound_value_error const& error) 
{
  std::string const* option_value =
    boost::get_error_info<errinfo_option_value>(error);

  std::stringstream ss;

  ss << "Found value without option";
  if (option_value != NULL)
  {
    ss << ": " << *option_value;
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::no_such_option_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);

  std::stringstream ss;

  ss << "Option does not exist";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << ": \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " in option namespace \"" << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << ": \"" << *option_flag << "\"";
  }

  size_t const* configuration_file_line =
    boost::get_error_info<mack::core::files::errinfo_parse_line>(error);

  std::string const* configuration_file =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  if (configuration_file_line != NULL || configuration_file != NULL)
  {
    ss << " from configuration file";
    if (configuration_file != NULL)
    {
      ss << " \"" << *configuration_file << "\"";
    }
    if (configuration_file_line != NULL)
    {
      ss << " in line " << configuration_file_line;
    }
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::option_collision_error const& error) 
{
  std::set<std::string> const* option_names =
    boost::get_error_info<errinfo_option_names>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);
  std::string const* option_value =
    boost::get_error_info<errinfo_option_value>(error);

  std::stringstream ss;

  ss << "Ambiguous option";

  if (option_flag != NULL)
  {
    ss << " \"" << *option_flag << '"';
  }

  if (option_value != NULL)
  {
    ss << " with assigned value \"" << *option_value << "\"";
  }

  if (option_names != NULL)
  {
    ss << " (candidates: ";
    bool first = true;

    for (std::set<std::string>::const_iterator names_it = option_names->begin();
        names_it != option_names->end();
        ++names_it)
    {
      if (first)
      {
        first = false;
      }
      else
      {
        ss << ", ";
      }
      ss << *names_it;
    }

    ss << ')';
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::no_value_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);

  std::stringstream ss;

  ss << "No value was provided";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << " for option \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " for a mandatory option in option namespace \""
        << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << " for option \"" << *option_flag << "\"";
  }
  else
  {
    ss << " for a mandatory option";
  }

  size_t const* configuration_file_line =
    boost::get_error_info<mack::core::files::errinfo_parse_line>(error);

  std::string const* configuration_file =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  if (configuration_file_line != NULL || configuration_file != NULL)
  {
    ss << " from configuration file";
    if (configuration_file != NULL)
    {
      ss << " \"" << *configuration_file << "\"";
    }
    if (configuration_file_line != NULL)
    {
      ss << " in line " << configuration_file_line;
    }
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::no_selection_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);

  std::stringstream ss;

  ss << "Option is not a selection";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << ": \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " in option namespace \""
        << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << ": \"" << *option_flag << "\"";
  }

  return ss.str();
}

std::string
mack::options::get_error_message(
    mack::options::no_such_namespace_error const& error) 
{
  std::string const* option_namespace =
    boost::get_error_info<errinfo_option_namespace>(error);
  std::string const* option_flag =
    boost::get_error_info<errinfo_option_flag>(error);

  std::stringstream ss;

  ss << "Option namespace does not exist";
  if (option_namespace != NULL && !(option_namespace->empty()))
  {
    if (option_flag != NULL)
    {
      ss << ": \"" << *option_namespace << option_namespace_separator
        << *option_flag << "\"";
    }
    else
    {
      ss << " in option namespace \""
        << *option_namespace << "\"";
    }
  }
  else if (option_flag != NULL)
  {
    ss << ": \"" << *option_flag << "\"";
  }

  size_t const* configuration_file_line =
    boost::get_error_info<mack::core::files::errinfo_parse_line>(error);

  std::string const* configuration_file =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  if (configuration_file_line != NULL || configuration_file != NULL)
  {
    ss << " from configuration file";
    if (configuration_file != NULL)
    {
      ss << " \"" << *configuration_file << "\"";
    }
    if (configuration_file_line != NULL)
    {
      ss << " in line " << configuration_file_line;
    }
  }

  return ss.str();
}

