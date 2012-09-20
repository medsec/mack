#include "stream_logger.hpp"

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>

const std::string mack::logging::stream_logger::flag_output = "output";

const std::string mack::logging::stream_logger::selection_output_stdout =
  "std::cout";

const std::string mack::logging::stream_logger::selection_output_stderr =
  "std::cerr";

mack::logging::stream_logger::stream_logger(
    mack::options::values const* values)
  : _output(&std::cout)
{
  std::string output_stream = values->get(flag_output);
  if (output_stream.compare(selection_output_stdout) == 0)
  {
    _output = &std::cout;
  }
  else if (output_stream.compare(selection_output_stderr) == 0)
  {
    _output = &std::cerr;
  }
  else
  {
    BOOST_THROW_EXCEPTION(mack::options::invalid_value_error()
        << mack::options::errinfo_option_value(output_stream)
        << mack::options::errinfo_option_flag(flag_output)
        << mack::options::errinfo_option_value_description(
          selection_output_stdout + " || " + selection_output_stderr));
  }
}

mack::logging::stream_logger::stream_logger(
    std::ostream& output_stream)
  : _output(&output_stream)
{
}

mack::logging::stream_logger::~stream_logger()
{
}

void
mack::logging::stream_logger::log(
    std::string const& message)
{
  (*_output) << message << std::endl;
}

