#include "program_options.hpp"

#include <mack/options/option.hpp>
#include <mack/options/option_switch.hpp>
#include <mack/options/selection_option.hpp>
#include <mack/options/type_option.hpp>

#include <mack/options/types/loggers.hpp>
#include <mack/logging/logger.hpp>
#include <mack/logging/stream_logger.hpp>
#include <mack/logging.hpp>

#include <mack/options/configuration_file.hpp>
#include <mack/options/commandline_formatter.hpp>

#include <mack/options/types/option_handlers.hpp>
#include <mack/options/handlers/handler.hpp>
#include <mack/options/handlers/no_interaction_handler.hpp>
#include <mack/options/handlers/basic_commandline_handler.hpp>

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>

#include <set>

#include <cctype>

const std::string help_short_flag = "h";
const std::string help_long_flag = "help";
const std::string configuration_file_short_flag = "c";
const std::string configuration_file_long_flag = "config";
const std::string logger_debug_short_flag = "dL";
const std::string logger_debug_long_flag = "debug_logger";
const std::string logger_info_short_flag = "iL";
const std::string logger_info_long_flag = "info_logger";
const std::string logger_warning_short_flag = "wL";
const std::string logger_warning_long_flag = "warning_logger";
const std::string logger_error_short_flag = "eL";
const std::string logger_error_long_flag = "error_logger";
const std::string verbosity_short_flag = "v";
const std::string verbosity_long_flag = "verbosity";
const std::string interaction_handler_short_flag = "i";
const std::string interaction_handler_long_flag = "interactive";

const std::string no_logger = "mack::logging::previous_logger";

const std::string loglevel_debug = "debug";
const std::string loglevel_info = "info";
const std::string loglevel_warning = "warning";
const std::string loglevel_error = "error";

mack::options::program_options::program_options(
    std::string const& name,
    std::string const& brief_description,
    std::string const& detailed_description,
    std::vector<option*> const& options)
  : mack::options::options(brief_description, detailed_description, options),
    _name(name),
    _help_option(
        mack::options::program_options::create_help_option()),
    _configuration_file_option(
        mack::options::program_options::create_configuration_file_option()),
    _debug_logger_option(
        mack::options::program_options::create_debug_logger_option()),
    _info_logger_option(
        mack::options::program_options::create_info_logger_option()),
    _warning_logger_option(
        mack::options::program_options::create_warning_logger_option()),
    _error_logger_option(
        mack::options::program_options::create_error_logger_option()),
    _verbosity_option(
        mack::options::program_options::create_verbosity_option()),
    _interaction_handler_option(
        mack::options::program_options::create_interaction_handler_option()),
    _commandline_arguments_done(false)
{
  std::set<std::string> default_flags;
  default_flags.insert(help_short_flag);
  default_flags.insert(help_long_flag);
  default_flags.insert(configuration_file_short_flag);
  default_flags.insert(configuration_file_long_flag);
  default_flags.insert(logger_debug_short_flag);
  default_flags.insert(logger_debug_long_flag);
  default_flags.insert(logger_info_short_flag);
  default_flags.insert(logger_info_long_flag);
  default_flags.insert(logger_warning_short_flag);
  default_flags.insert(logger_warning_long_flag);
  default_flags.insert(logger_error_short_flag);
  default_flags.insert(logger_error_long_flag);
  default_flags.insert(verbosity_short_flag);
  default_flags.insert(verbosity_long_flag);
  default_flags.insert(interaction_handler_short_flag);
  default_flags.insert(interaction_handler_long_flag);

  for (std::vector<option*>::iterator options_it = begin();
      options_it != end();
      ++options_it)
  {
    if (default_flags.count((*options_it)->get_short_flag()) > 0)
    {
      BOOST_THROW_EXCEPTION(flag_collision_error()
          << errinfo_option_flag((*options_it)->get_short_flag())
          << errinfo_option_namespace(""));
    }
    if (default_flags.count((*options_it)->get_long_flag()) > 0)
    {
      BOOST_THROW_EXCEPTION(flag_collision_error()
          << errinfo_option_flag((*options_it)->get_long_flag())
          << errinfo_option_namespace(""));
    }
  }
}

mack::options::program_options::~program_options()
{
  delete _help_option;
  delete _configuration_file_option;
  delete _debug_logger_option;
  delete _info_logger_option;
  delete _warning_logger_option;
  delete _error_logger_option;
  delete _verbosity_option;
  delete _interaction_handler_option;
}

std::string const&
mack::options::program_options::get_program_name() const
{
  return _name;
}

void
mack::options::program_options::set(
    std::string const& flag)
{
  std::string option_flag = flag; // flag without namespaces
  const std::vector<std::string> namespaces = separate_namespaces(option_flag);
  set(namespaces, option_flag);
}

void
mack::options::program_options::set(
    std::vector<std::string> const& namespaces,
    std::string const& flag)
{
  option* opt = find_option(namespaces, flag);
  if (opt == _configuration_file_option) // configuration file requires value
  {
    BOOST_THROW_EXCEPTION(no_value_error()
        << errinfo_option_namespace("")
        << errinfo_option_flag(opt->get_long_flag()));
  }
  opt->set_value();
}

void
mack::options::program_options::set(
    std::string const& flag,
    std::string const& value)
{
  std::string option_flag = flag; // flag without namespaces
  const std::vector<std::string> namespaces = separate_namespaces(option_flag);
  set(namespaces, option_flag, value);
}

void
mack::options::program_options::set(
    std::vector<std::string> const& namespaces,
    std::string const& flag,
    std::string const& value)
{
  option* opt = NULL;
  try
  {
    opt = find_option(namespaces, flag);
  }
  catch (no_such_option_error e)
  {
    e << errinfo_option_value(value);
    throw;
  }
  catch (option_collision_error e)
  {
    e << errinfo_option_value(value);
    throw;
  }
  catch (no_such_namespace_error e)
  {
    e << errinfo_option_value(value);
    throw;
  }

  if (opt == _configuration_file_option)
  {
    set_from_configuration_file(boost::filesystem::path(value));
  }
  else
  {
    opt->set_value(value);
  }
}

void
mack::options::program_options::set_from_configuration(
    std::string const& configuration)
{
  mack::options::configuration_file::parse(configuration, this);
}

void
mack::options::program_options::set_from_configuration_file(
    boost::filesystem::path const& file_path)
{
  mack::options::configuration_file::parse(file_path, this);
}

void
mack::options::program_options::set_all(
    const int argc,
    char const* const* argv,
    const int offset)
{
  int i = offset;
  while (i < argc)
  {
    const std::string v(argv[i]);
    // require an option start (-- or -)
    if (v[0] != '-')
    {
      BOOST_THROW_EXCEPTION(unbound_value_error()
          << errinfo_option_value(v));
    }
    std::string flag;
    if (v.size() > 1 && v[1] == '-')
    {
      flag = v.substr(2);
    }
    else
    {
      flag = v.substr(1);
    }

    // check if flag contains the value (--flag=value)
    const size_t equals_index = flag.find('=');
    if (equals_index == std::string::npos)
    {
      // check if next element is a value
      if (i + 1 == argc || argv[i + 1][0] == '-')
      {
        // next element is a new option/end => this is a switch
        set(flag);
      }
      else
      {
        // next element is the value
        set(flag, std::string(argv[i + 1]));
        ++i; // increment additionally
      }
    }
    else // (equals_index != std::string::npos)
    {
      // get value from flag
      const std::string value = flag.substr(equals_index + 1);
      flag = flag.substr(0, equals_index);
      set(flag, value);
    }
    ++i;
  }
}

void
mack::options::program_options::set_commandline_arguments_done()
{
  _commandline_arguments_done = true;
}

mack::options::option const*
mack::options::program_options::get_option(
    std::string const& flag) const
{
  if (!_commandline_arguments_done && (flag == help_short_flag
      || flag == help_long_flag))
  {
    return _help_option;
  }
  else if (!_commandline_arguments_done
      && (flag == interaction_handler_short_flag
        || flag == interaction_handler_long_flag))
  {
    return _interaction_handler_option;
  }
  else if (flag == configuration_file_short_flag
        || flag == configuration_file_long_flag)
  {
    return _configuration_file_option;
  }
  else if (flag == logger_debug_short_flag 
        || flag == logger_debug_long_flag)
  {
    return _debug_logger_option;
  }
  else if (flag == logger_info_short_flag 
        || flag == logger_info_long_flag)
  {
    return _info_logger_option;
  }
  else if (flag == logger_warning_short_flag
        || flag == logger_warning_long_flag)
  {
    return _warning_logger_option;
  }
  else if (flag == logger_error_short_flag 
        || flag == logger_error_long_flag)
  {
    return _error_logger_option;
  }
  else if (flag == verbosity_short_flag 
        || flag == verbosity_long_flag)
  {
    return _verbosity_option;
  }
  return mack::options::options::get_option(flag);
}

mack::options::option*
mack::options::program_options::get_option(
    std::string const& flag)
{
  if (!_commandline_arguments_done && (flag == help_short_flag
      || flag == help_long_flag))
  {
    return _help_option;
  }
  else if (!_commandline_arguments_done
      && (flag == interaction_handler_short_flag
        || flag == interaction_handler_long_flag))
  {
    return _interaction_handler_option;
  }
  else if (flag == configuration_file_short_flag
        || flag == configuration_file_long_flag)
  {
    return _configuration_file_option;
  }
  else if (flag == logger_debug_short_flag 
        || flag == logger_debug_long_flag)
  {
    return _debug_logger_option;
  }
  else if (flag == logger_info_short_flag 
        || flag == logger_info_long_flag)
  {
    return _info_logger_option;
  }
  else if (flag == logger_warning_short_flag
        || flag == logger_warning_long_flag)
  {
    return _warning_logger_option;
  }
  else if (flag == logger_error_short_flag 
        || flag == logger_error_long_flag)
  {
    return _error_logger_option;
  }
  else if (flag == verbosity_short_flag 
        || flag == verbosity_long_flag)
  {
    return _verbosity_option;
  }
  return mack::options::options::get_option(flag);
}

mack::options::values const*
mack::options::program_options::create() const
{
  if (is_help_set())
  {
    return NULL;
  }
  else
  {
    mack::logging::clear_loggers();

    value const* debug_logger_value = _debug_logger_option->create();
    if (debug_logger_value->get_value_class_name().compare(no_logger) != 0)
    {
      mack::logging::set_debug_logger(
          debug_logger_value->get<mack::logging::logger>());
    }
    delete debug_logger_value;

    value const* info_logger_value = _info_logger_option->create();
    if (info_logger_value->get_value_class_name().compare(no_logger) != 0)
    {
      mack::logging::set_info_logger(
          info_logger_value->get<mack::logging::logger>());
    }
    delete info_logger_value;

    value const* warning_logger_value = _warning_logger_option->create();
    if (warning_logger_value->get_value_class_name().compare(no_logger) != 0)
    {
      mack::logging::set_warning_logger(
          warning_logger_value->get<mack::logging::logger>());
    }
    delete warning_logger_value;

    value const* error_logger_value = _error_logger_option->create();
    if (error_logger_value->get_value_class_name().compare(no_logger) != 0)
    {
      mack::logging::set_error_logger(
          error_logger_value->get<mack::logging::logger>());
    }
    delete error_logger_value;

    value const* verbosity_value = _verbosity_option->create();
    const std::string loglevel = verbosity_value->get();
    if (loglevel == loglevel_debug)
    {
      mack::logging::set_log_level_to_debug();
    }
    else if (loglevel == loglevel_info)
    {
      mack::logging::set_log_level_to_info();
    }
    else if (loglevel == loglevel_warning)
    {
      mack::logging::set_log_level_to_warning();
    }
    else
    {
      mack::logging::set_log_level_to_error();
    }
    delete verbosity_value;

    return mack::options::options::create();
  }
}

bool
mack::options::program_options::is_help_set() const
{
  value const* help_value = _help_option->create();
  const bool help = help_value->get_boolean();
  delete help_value;
  return help;
}

void
mack::options::program_options::print_help(
    std::ostream& output_stream) const
{
  commandline_formatter formatter(output_stream, true, false);

  output_stream << ' ' << get_program_name() << std::endl;
  std::string underlining;
  underlining.append(get_program_name().length() + 2, '=');
  output_stream << underlining << std::endl;

  formatter.print(this);

  output_stream << std::endl << "Default Options: " << std::endl;
  formatter.print_brief(_help_option);
  formatter.print_brief(_interaction_handler_option);
  std::vector<option const*> default_options = get_default_options();
  for (std::vector<option const*>::const_iterator options_it =
        default_options.begin();
      options_it != default_options.end();
      ++options_it)
  {
    formatter.print_brief(*options_it);
  }
}

mack::options::handlers::handler*
mack::options::program_options::create_interaction_handler() const
{
  value const* handler_value = _interaction_handler_option->create();
  mack::options::handlers::handler* option_handler =
    handler_value->get<mack::options::handlers::handler>();
  delete handler_value;
  return option_handler;
}

std::vector<mack::options::option const*>
mack::options::program_options::get_default_options() const
{
  std::vector<option const*> default_options;
  default_options.push_back(get_configuration_file_option());
  default_options.push_back(get_debug_logger_option());
  default_options.push_back(get_info_logger_option());
  default_options.push_back(get_warning_logger_option());
  default_options.push_back(get_error_logger_option());
  default_options.push_back(get_verbosity_option());
  return default_options;
}

std::vector<mack::options::option*>
mack::options::program_options::get_default_options()
{
  std::vector<option*> default_options;
  default_options.push_back(_configuration_file_option);
  default_options.push_back(_debug_logger_option);
  default_options.push_back(_info_logger_option);
  default_options.push_back(_warning_logger_option);
  default_options.push_back(_error_logger_option);
  default_options.push_back(_verbosity_option);
  return default_options;
}

mack::options::option const*
mack::options::program_options::get_configuration_file_option() const
{
  return _configuration_file_option;
}

mack::options::option const*
mack::options::program_options::get_debug_logger_option() const
{
  return _debug_logger_option;
}

mack::options::option const*
mack::options::program_options::get_info_logger_option() const
{
  return _info_logger_option;
}

mack::options::option const*
mack::options::program_options::get_warning_logger_option() const
{
  return _warning_logger_option;
}

mack::options::option const*
mack::options::program_options::get_error_logger_option() const
{
  return _error_logger_option;
}

mack::options::option const*
mack::options::program_options::get_verbosity_option() const
{
  return _verbosity_option;
}

void
mack::options::program_options::check_valid_flag(
    std::string const& flag) const
{
  // minimum of one character
  if (flag.length() == 0)
  {
    BOOST_THROW_EXCEPTION(invalid_flag_error()
        << errinfo_option_flag(flag));
  }

  // first character alphanumeric
  if (!isalnum(flag[0]))
  {
    BOOST_THROW_EXCEPTION(invalid_flag_error()
        << errinfo_option_flag(flag));
  }

  for (size_t i = 1; i < flag.length(); ++i)
  {
    // remaining characters alphanumeric, '_' or '-'
    if (!isalnum(flag[0]) && flag[0] != '_' && flag[0] != '-')
    {
      BOOST_THROW_EXCEPTION(invalid_flag_error()
          << errinfo_option_flag(flag));
    }
  }
}

std::vector<std::string>
mack::options::program_options::separate_namespaces(
    std::string& flag) const
{
  std::vector<std::string> namespaces;
  size_t namespace_start = 0;
  size_t separator_index = flag.find(option_namespace_separator);
  while (separator_index != std::string::npos)
  {
    namespaces.push_back(
        flag.substr(namespace_start, separator_index - namespace_start));
    namespace_start = separator_index + 1;
    separator_index = flag.find(option_namespace_separator, namespace_start);
  }
  flag = flag.substr(namespace_start);
  return namespaces;
}

mack::options::option*
mack::options::program_options::find_option(
    std::vector<std::string> const& namespaces,
    std::string const& flag)
{
  check_valid_flag(flag);

  if (namespaces.empty())
  {
    // check default options
    if (!_commandline_arguments_done)
    {
      // interaction handler and help are still available
      if (flag == help_short_flag
          || flag == help_long_flag)
      {
        return _help_option;
      }
      if (flag == interaction_handler_short_flag
          || flag == interaction_handler_long_flag)
      {
        return _interaction_handler_option;
      }
    }
    std::vector<option*> default_options = get_default_options();
    for (std::vector<option*>::iterator default_options_it =
          default_options.begin();
        default_options_it != default_options.end();
        ++default_options_it)
    {
      if (flag == (*default_options_it)->get_short_flag()
          || flag == (*default_options_it)->get_long_flag())
      {
        return *default_options_it;
      }
    }

    // check default namespace options
    for (std::vector<option*>::iterator options_it = begin();
        options_it != end();
        ++options_it)
    {
      if (flag == (*options_it)->get_short_flag()
          || flag == (*options_it)->get_long_flag())
      {
        return *options_it;
      }
    }

    // check all options (collisions?)
    std::set<std::pair<std::string, option*> > collisions =
       search_for_option(flag, "");
    if (collisions.size() == 1) // unambiguous
    {
      return collisions.begin()->second;
    }
    else if (collisions.size() > 1) // ambiguous
    {
      std::set<std::string> colliding_options;
      for (std::set<std::pair<std::string, option*> >::iterator collisions_it =
            collisions.begin();
          collisions_it != collisions.end();
          ++collisions_it)
      {
        if (collisions_it->first.empty())
        {
          colliding_options.insert(
              collisions_it->second->get_long_flag());
        }
        else
        {
          colliding_options.insert(
              collisions_it->first + option_namespace_separator
                + collisions_it->second->get_long_flag());
        }
      }
      BOOST_THROW_EXCEPTION(option_collision_error()
          << errinfo_option_names(colliding_options)
          << errinfo_option_flag(flag));
    }

    // none found
    BOOST_THROW_EXCEPTION(no_such_option_error()
        << errinfo_option_flag(flag));
  }
  else // !namespaces.empty()
  {
    options* opts = this;
    std::string namespace_total;
    std::vector<std::string>::const_iterator namespaces_it;
    // get to options described by namespaces
    for (namespaces_it = namespaces.begin();
        namespaces_it != namespaces.end();
        ++namespaces_it)
    {
      option* opt;
      try
      {
        opt = opts->get_option(*namespaces_it);
      }
      catch (no_such_option_error e) // no such option
      {
        BOOST_THROW_EXCEPTION(no_such_namespace_error()
            << errinfo_option_namespace(namespace_total)
            << errinfo_option_flag(*namespaces_it));
      }

      if (!opt->has_child_options())
      {
        // ERROR
        ++namespaces_it;
        if (namespaces_it == namespaces.end())
        {
          BOOST_THROW_EXCEPTION(no_such_option_error()
              << errinfo_option_namespace(namespace_total)
              << errinfo_option_flag(flag));
        }
        else
        {
          BOOST_THROW_EXCEPTION(no_such_namespace_error()
              << errinfo_option_namespace(namespace_total)
              << errinfo_option_flag(*namespaces_it));
        }
      }
      opts = opt->get_child_options();
      if (!namespace_total.empty())
      {
        namespace_total.append(option_namespace_separator);
      }
      namespace_total.append(*namespaces_it);
    }

    try
    {
      // return the option
      return opts->get_option(flag);
    }
    catch (no_such_option_error e)
    {
      e << errinfo_option_namespace(namespace_total);
      throw;
    }
  }
}

mack::options::option*
mack::options::program_options::create_help_option()
{
  option* option = new option_switch(
      help_short_flag,
      help_long_flag,
      "Prints help on this program and exits.",
      "");
  return option;
}

mack::options::option*
mack::options::program_options::create_configuration_file_option()
{
  option* option = new mack::options::option(
      configuration_file_short_flag,
      configuration_file_long_flag,
      "Add options from a configuration file.",
      "");
  return option;
}

mack::options::option*
mack::options::program_options::create_debug_logger_option()
{
  option* option = new type_option(
      logger_debug_short_flag,
      logger_debug_long_flag,
      "Sets the logger to be used for logging debug messages.",
      std::string("Will also be used for info, warning and error messages ")
      + "if they are set to mack::logging::previous_logger. "
      + "See also --" + verbosity_long_flag + ".",
      false,
      new mack::options::types::loggers(),
      "mack::logging::stream_logger");
  option->get_child_option(mack::logging::stream_logger::flag_output)->set_value(
      mack::logging::stream_logger::selection_output_stdout);
  return option;
}

mack::options::option*
mack::options::program_options::create_info_logger_option()
{
  option* option = new type_option(
      logger_info_short_flag,
      logger_info_long_flag,
      "Sets the logger to be used for logging info messages.",
      std::string("Will also be used for warning and error messages ")
      + "if they are set to mack::logging::previous_logger. "
      + "See also --" + verbosity_long_flag + ".",
      false,
      new mack::options::types::loggers(),
      no_logger);
  return option;
}

mack::options::option*
mack::options::program_options::create_warning_logger_option()
{
  option* option = new type_option(
      logger_warning_short_flag,
      logger_warning_long_flag,
      "Sets the logger to be used for logging warning messages.",
      std::string("Will also be used for error messages ")
      + "if they the error logger is set to mack::logging::previous_logger. "
      + "See also --" + verbosity_long_flag + ".",
      false,
      new mack::options::types::loggers(),
      "mack::logging::stream_logger");
  option->get_child_option(mack::logging::stream_logger::flag_output)->set_value(
      mack::logging::stream_logger::selection_output_stderr);
  return option;
}

mack::options::option*
mack::options::program_options::create_error_logger_option()
{
  option* option = new type_option(
      logger_error_short_flag,
      logger_error_long_flag,
      "Sets the logger to be used for logging error messages.",
      std::string("See also --") + verbosity_long_flag + ".",
      false,
      new mack::options::types::loggers(),
      no_logger);
  return option;
}

mack::options::option*
mack::options::program_options::create_verbosity_option()
{
  std::vector<std::string> values;
  values.push_back(loglevel_debug);
  values.push_back(loglevel_info);
  values.push_back(loglevel_warning);
  values.push_back(loglevel_error);
  option* option = new selection_option(
      verbosity_short_flag,
      verbosity_long_flag,
      "Sets the minimum log level required for a message to be displayed.",
      std::string("See also --") + logger_debug_long_flag + ".",
      values,
      loglevel_info);
  return option;
}

mack::options::option*
mack::options::program_options::create_interaction_handler_option()
{
  option* option = new type_option(
      interaction_handler_short_flag,
      interaction_handler_long_flag,
      "Sets the interaction handler to be used.",
      std::string("If an interaction handler is set, program options can be")
      + "specified interactively on a prompt.",
      false,
      new mack::options::types::option_handlers(),
      "mack::options::handlers::basic_commandline_handler",
      "mack::options::handlers::no_interaction_handler");
  return option;
}

std::set<std::pair<std::string, mack::options::option*> >
mack::options::program_options::search_for_option(
    std::string const& flag,
    std::string const& this_namespace)
{
  std::set<std::pair<std::string, option*> > found;
  if (!_commandline_arguments_done && (flag == help_short_flag
      || flag == help_long_flag))
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _help_option));
  }
  else if (!_commandline_arguments_done
      && (flag == interaction_handler_short_flag
        || flag == interaction_handler_long_flag))
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _interaction_handler_option));
  }
  else if (flag == configuration_file_short_flag
        || flag == configuration_file_long_flag)
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _configuration_file_option));
  }
  else if (flag == logger_debug_short_flag 
        || flag == logger_debug_long_flag)
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _debug_logger_option));
  }
  else if (flag == logger_info_short_flag 
        || flag == logger_info_long_flag)
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _info_logger_option));
  }
  else if (flag == logger_warning_short_flag
        || flag == logger_warning_long_flag)
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _warning_logger_option));
  }
  else if (flag == logger_error_short_flag 
        || flag == logger_error_long_flag)
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _error_logger_option));
  }
  else if (flag == verbosity_short_flag 
        || flag == verbosity_long_flag)
  {
    found.insert(std::pair<std::string, option*>(
          this_namespace, _verbosity_option));
  }

  if (!_commandline_arguments_done && _help_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _help_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _help_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (!_commandline_arguments_done
      && _interaction_handler_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _interaction_handler_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _interaction_handler_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (_configuration_file_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _configuration_file_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _configuration_file_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (_debug_logger_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _debug_logger_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _debug_logger_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (_info_logger_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _info_logger_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _info_logger_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (_warning_logger_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _warning_logger_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _warning_logger_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (_error_logger_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _error_logger_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _error_logger_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }
  if (_verbosity_option->has_child_options())
  {
    std::set<std::pair<std::string, option*> > found_recursive =
      _verbosity_option->get_child_options()->search_for_option(
          flag,
          this_namespace + option_namespace_separator
            + _verbosity_option->get_long_flag());
    found.insert(found_recursive.begin(), found_recursive.end());
  }

  std::set<std::pair<std::string, option*> > found_custom =
    mack::options::options::search_for_option(flag, this_namespace);
  found.insert(found_custom.begin(), found_custom.end());
  return found;
}

