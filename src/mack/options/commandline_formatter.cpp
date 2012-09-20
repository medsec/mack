#include "commandline_formatter.hpp"

#include <boost/algorithm/string/trim.hpp>

#include <vector>

mack::options::commandline_formatter::commandline_formatter(
    std::ostream& output_stream,
    const bool print_flag_prefix,
    const bool print_full_descriptions,
    const size_t line_length_chars,
    const size_t tabsize)
  : _output_stream(output_stream),
    _print_flag_prefixes(print_flag_prefix),
    _print_full_descriptions(print_full_descriptions),
    _line_length_chars(line_length_chars),
    _tabsize(tabsize),
    _indent(),
    _already_filled(0)
{
}

mack::options::commandline_formatter::~commandline_formatter()
{
}

void
mack::options::commandline_formatter::print(
    mack::options::option const* option,
    std::string const& option_namespace)
{
  print_option_name(option, option_namespace);

  increase_indent(2);
  print_attributes(option);
  print_description(option);
  decrease_indent();

  print_children(option, option_namespace);
  decrease_indent();
}

void
mack::options::commandline_formatter::print_brief(
    mack::options::option const* option,
    std::string const& option_namespace)
{
  print_option_name(option, option_namespace, true);
  increase_indent(2);
  if (option->has_child_options())
  {
    if (!option_namespace.empty())
    {
      print_brief(
          option->get_child_options(),
          option_namespace + option_namespace_separator
            + option->get_long_flag());
    }
    else
    {
      print_brief(option->get_child_options(), option->get_long_flag());
    }
  }
  decrease_indent(2);
}

void
mack::options::commandline_formatter::print(
    mack::options::options const* options,
    std::string const& option_namespace)
{
  if (_print_full_descriptions)
  {
    print_line(boost::algorithm::trim_copy(options->get_full_description()));
  }
  else
  {
    print_line(boost::algorithm::trim_copy(options->get_brief_description()));
  }

  for (std::vector<option const*>::const_iterator options_it = options->begin();
      options_it != options->end();
      ++options_it)
  {
    print_line();
    print(*options_it, option_namespace);
  }
}

void
mack::options::commandline_formatter::print_brief(
    mack::options::options const* options,
    std::string const& option_namespace)
{
  for (std::vector<option const*>::const_iterator options_it = options->begin();
      options_it != options->end();
      ++options_it)
  {
    print_brief(*options_it, option_namespace);
  }
}

void
mack::options::commandline_formatter::print_description(
    mack::options::option const* option)
{
  if (_print_full_descriptions)
  {
    print_line(boost::algorithm::trim_copy(option->get_full_description()));
  }
  else
  {
    print_line(boost::algorithm::trim_copy(option->get_brief_description()));
  }
}

void
mack::options::commandline_formatter::print_description(
    std::string const& program_name,
    std::string const& brief_description,
    std::string const& full_description)
{
  print_line(boost::algorithm::trim_copy(program_name));
  _output_stream << std::endl;
  if (_print_full_descriptions)
  {
    print_line(boost::algorithm::trim_copy(full_description));
  }
  else
  {
    print_line(boost::algorithm::trim_copy(brief_description));
  }
}

void
mack::options::commandline_formatter::print_description(
    mack::options::options const* options)
{
  if (_print_full_descriptions)
  {
    print_line(boost::algorithm::trim_copy(options->get_full_description()));
  }
  else
  {
    print_line(boost::algorithm::trim_copy(options->get_brief_description()));
  }
}

void
mack::options::commandline_formatter::print_option_name(
    mack::options::option const* option,
    std::string const& option_namespace,
    const bool with_value)
{
  // short flag
  std::string short_flag;
  if (_print_flag_prefixes)
  {
    short_flag.append("-");
  }
  short_flag.append(boost::algorithm::trim_copy(option->get_short_flag()));
  short_flag.append(", ");
  print(short_flag);

  // long flag
  std::string long_flag;
  if (_print_flag_prefixes)
  {
    long_flag.append("--");
  }
  if (!option_namespace.empty())
  {
    long_flag.append(boost::algorithm::trim_copy(option_namespace));
    long_flag.append(option_namespace_separator);
  }
  long_flag.append(boost::algorithm::trim_copy(option->get_long_flag()));

  _output_stream << long_flag;
  if (with_value && option->has_value())
  {
    _output_stream << "=" << option->get_value();
  }

  _output_stream << std::endl;
  _already_filled = 0;
}

void
mack::options::commandline_formatter::print_attributes(
    mack::options::option const* option)
{
  // current value
  print(std::string("value  : "));
  increase_indent();
  if (option->has_value())
  {
    print_line(boost::algorithm::trim_copy(option->get_value()));
  }
  else
  {
    print_line(std::string("MISSING"));
  }
  decrease_indent();

  // default value
  if (option->has_default_value())
  {
    print(std::string("default: "));
    increase_indent();
    print_line(boost::algorithm::trim_copy(std::string(
            option->get_default_value())));
    decrease_indent();
  }

  // selection
  if (option->is_selection())
  {
    print_line("possible values:");
    increase_indent();
    for (std::vector<std::string>::const_iterator values_it =
          option->selection_values_begin();
        values_it != option->selection_values_end();
        ++values_it)
    {
      print("- ");
      _output_stream << boost::algorithm::trim_copy(*values_it) << std::endl;
      _already_filled = 0;
    }
    decrease_indent();
  }
}

void
mack::options::commandline_formatter::print_children(
    mack::options::option const* option,
    std::string const& option_namespace)
{
  if (option->has_child_options())
  {
    if (!option_namespace.empty())
    {
      print(
          option->get_child_options(),
          option_namespace + option_namespace_separator
            + option->get_long_flag());
    }
    else
    {
      print(option->get_child_options(), option->get_long_flag());
    }
  }
}

void
mack::options::commandline_formatter::increase_indent(
    const size_t amount)
{
  _indent.append(_tabsize * amount, ' ');
}

void
mack::options::commandline_formatter::decrease_indent(
    const size_t amount)
{
  _indent.erase(0, _tabsize * amount);
}

void
mack::options::commandline_formatter::print(
    std::string const& text)
{
  size_t remaining;
  if (_already_filled == 0)
  {
    // New line
    _output_stream << _indent;
    _already_filled = _indent.length();
    if (_indent.length() > _line_length_chars)
    {
      remaining = 0;
    }
    else
    {
      remaining = _line_length_chars - _indent.length();
    }
  }
  else
  {
    // Line was already begun
    if (_already_filled >= _line_length_chars)
    {
      _already_filled = 0;
      _output_stream << std::endl;
      print(text);
      return;
    }
    else
    {
      remaining = _line_length_chars - _already_filled;
    }
  }

  if (remaining >= text.length())
  {
    _output_stream << text;
    _already_filled += text.length();
  }
  else
  {
    const size_t linebreak_index = text.find('\n');
    if (linebreak_index != std::string::npos
        && linebreak_index < remaining)
    {
      // linebreak character before calculated linebreak
      _output_stream << text.substr(0, linebreak_index);
      _output_stream << std::endl;
      _already_filled = 0;
      print(text.substr(linebreak_index + 1));
    }
    else
    {
      const size_t last_space_index = text.find_last_of(" \t\n", remaining);
      if (last_space_index == std::string::npos)
      {
        if (_already_filled > _indent.length())
        {
          _already_filled = 0;
          _output_stream << std::endl;
          print(text);
          return;
        }

        // print at least one word
        const size_t first_space_index = text.find_first_of(" \t\n");
        if (first_space_index == std::string::npos)
        {
          // only one word left
          _output_stream << text;
          _output_stream << std::endl;
          _already_filled = 0;
        }
        else
        {
          _output_stream << text.substr(0, first_space_index);
          _output_stream << std::endl;
          _already_filled = 0;
          print(text.substr(first_space_index + 1));
        }
      }
      else
      {
        _output_stream << text.substr(0, last_space_index);
        _output_stream << std::endl;
        _already_filled = 0;
        print(text.substr(last_space_index + 1));
      }
    }
  }
}

void
mack::options::commandline_formatter::print_line(
    std::string const& text)
{
  if (text.empty())
  {
    _output_stream << std::endl;
    _already_filled = 0;
  }
  else
  {
    print(text);
    if (_already_filled != 0)
    {
      _output_stream << std::endl;
      _already_filled = 0;
    }
  }
}

