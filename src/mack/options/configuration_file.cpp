#include "configuration_file.hpp"

#include <mack/core/files.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/exception/get_error_info.hpp>
#include <string>
#include <mack/core/null_pointer_error.hpp>

void
do_parse_line(
    std::string const& line,
    mack::options::program_options* options,
    const size_t line_number)
{
  try
  {
    mack::options::configuration_file::parse_line(line, options);
  }
  catch (mack::options::no_such_option_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_parse_line>(e) == NULL)
    {
      // this is the configuration file with the erroneous line
      e << mack::core::files::errinfo_parse_line(line_number);
    } // else error from within a further configuration file
    throw;
  }
  catch (mack::options::no_such_namespace_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_parse_line>(e) == NULL)
    {
      e << mack::core::files::errinfo_parse_line(line_number);
    }
    throw;
  }
  catch (mack::options::invalid_value_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_parse_line>(e) == NULL)
    {
      e << mack::core::files::errinfo_parse_line(line_number);
    }
    throw;
  }
  catch (mack::options::no_value_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_parse_line>(e) == NULL)
    {
      e << mack::core::files::errinfo_parse_line(line_number);
    }
    throw;
  }
}

void
mack::options::configuration_file::parse(
    boost::filesystem::path const& file_path,
    mack::options::program_options* options)
{
  if (options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  try
  {
    parse(mack::core::files::read_file(file_path), options);
  }
  catch (mack::options::no_such_option_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_file>(e) == NULL)
    {
      // this is the configuration file with the erroneous line
      e << mack::core::files::errinfo_file(file_path.native());
    } // else error from within a further configuration file
    throw;
  }
  catch (mack::options::no_such_namespace_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_file>(e) == NULL)
    {
      e << mack::core::files::errinfo_file(file_path.native());
    }
    throw;
  }
  catch (mack::options::invalid_value_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_file>(e) == NULL)
    {
      e << mack::core::files::errinfo_file(file_path.native());
    }
    throw;
  }
  catch (mack::options::no_value_error e)
  {
    if (boost::get_error_info<mack::core::files::errinfo_file>(e) == NULL)
    {
      e << mack::core::files::errinfo_file(file_path.native());
    }
    throw;
  }
}

void
mack::options::configuration_file::parse(
    std::string const& content,
    mack::options::program_options* options)
{
  if (options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  size_t line_number = 0;
  size_t line_start = 0;
  size_t line_end = content.find('\n', line_start);
  while (line_end != std::string::npos)
  {
    do_parse_line(
        content.substr(line_start, line_end - line_start), options,
        line_number);
    // prepare for next line
    ++line_number;
    line_start = line_end + 1;
    line_end = content.find('\n', line_start);
  }
  do_parse_line(
      content.substr(line_start), options,
      line_number);
}

void
mack::options::configuration_file::parse_line(
    std::string const& line,
    mack::options::program_options* options)
{
  if (options == NULL)
  {
    BOOST_THROW_EXCEPTION(mack::core::null_pointer_error());
  }
  std::string trimmed_line = boost::algorithm::trim_copy(line);
  // Check for comment
  const size_t comment_index = trimmed_line.find('#');
  if (comment_index != std::string::npos)
  {
    trimmed_line = trimmed_line.substr(0, comment_index);
    boost::algorithm::trim(trimmed_line);
  }

  if (!trimmed_line.empty())
  {
    const size_t equals_index = trimmed_line.find('=');
    if (equals_index == std::string::npos)
    {
      // check for space
      const size_t space_index = trimmed_line.find_first_of(" \t");
      if (space_index == std::string::npos)
      {
        options->set(trimmed_line);
      }
      else
      {
        // get value and flag
        const std::string flag =
          boost::algorithm::trim_copy(trimmed_line.substr(0, space_index));
        const std::string value =
          boost::algorithm::trim_copy(trimmed_line.substr(space_index + 1));
        options->set(flag, value);
      }
    }
    else // (equals_index != std::string::npos)
    {
      // get value and flag
      const std::string flag =
        boost::algorithm::trim_copy(trimmed_line.substr(0, equals_index));
      const std::string value =
        boost::algorithm::trim_copy(trimmed_line.substr(equals_index + 1));
      options->set(flag, value);
    }
  }
}

