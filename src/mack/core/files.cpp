#include "files.hpp"

#include <iostream>
#include <fstream>

#include <boost/throw_exception.hpp>

#include <boost/exception/get_error_info.hpp>

std::string
mack::core::files::read_file(
    boost::filesystem::path const& file_path)
{
  // checking
  if (!boost::filesystem::exists(file_path)) {
    BOOST_THROW_EXCEPTION(file_not_exists_error()
        << errinfo_file(file_path.native()));
  }
  if (!boost::filesystem::is_regular_file(file_path)) {
    BOOST_THROW_EXCEPTION(not_a_file_error()
        << errinfo_file(file_path.native()));
  }

  // open stream
  std::ifstream file_stream;
  file_stream.open(file_path.c_str(), std::ios_base::in);
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }

  // getting length
  file_stream.seekg(0, std::ios::end);
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }
  const int length = file_stream.tellg();
  if (length < 0) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }
  file_stream.seekg(0, std::ios::beg);
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }

  // reading
  char* buffer = new char[length];
  file_stream.read(buffer, length);
  if (file_stream.fail()) {
    delete[] buffer;
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }
  std::string content(buffer, length);

  // cleaning
  file_stream.close();
  delete[] buffer;

  return content;
}

void
mack::core::files::write_file(
    std::string const& content,
    boost::filesystem::path const& file_path)
{
  // checking
  const boost::filesystem::path parent = file_path.parent_path();
  if (!boost::filesystem::exists(parent)) {
    BOOST_THROW_EXCEPTION(file_not_exists_error()
        << errinfo_file(parent.native()));
  }
  if (!boost::filesystem::is_directory(parent)) {
    BOOST_THROW_EXCEPTION(not_a_directory_error()
        << errinfo_file(parent.native()));
  }

  std::ofstream file_stream(file_path.c_str());
  file_stream << content;
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_write_error()
        << errinfo_file(file_path.native()));
  }

  file_stream.close();
}

bool
mack::core::files::is_content_of(
    boost::filesystem::path const& file_path,
    std::string const& content)
{
  // checking
  if (!boost::filesystem::exists(file_path)) {
    BOOST_THROW_EXCEPTION(file_not_exists_error()
        << errinfo_file(file_path.native()));
  }
  if (!boost::filesystem::is_regular_file(file_path)) {
    BOOST_THROW_EXCEPTION(not_a_file_error()
        << errinfo_file(file_path.native()));
  }

  // open stream
  std::ifstream file_stream;
  file_stream.open(file_path.c_str(), std::ios_base::in);
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }

  // getting length
  file_stream.seekg(0, std::ios::end);
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }
  const int length = file_stream.tellg();
  if (length < 0) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }

  // length missmatch?
  if (((unsigned int) length) != content.length()) {
    file_stream.close();
    return false;
  }

  file_stream.seekg(0, std::ios::beg);
  if (file_stream.fail()) {
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }

  // reading
  char* buffer = new char[length];
  file_stream.read(buffer, length);
  if (file_stream.fail()) {
    delete[] buffer;
    file_stream.close();
    BOOST_THROW_EXCEPTION(file_read_error()
        << errinfo_file(file_path.native()));
  }

  const std::string buffer_content = std::string(buffer, length);

  delete[] buffer;
  file_stream.close();

  return content == buffer_content;
}

std::string
mack::core::files::get_error_message(
    mack::core::files::file_not_exists_error const& error) 
{
  std::string const* file_name =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  std::stringstream ss;

  ss << "File does not exist";
  if (file_name != NULL)
  {
    ss << ": \"" << *file_name << "\"";
  }

  return ss.str();
}

std::string
mack::core::files::get_error_message(
    mack::core::files::not_a_file_error const& error) 
{
  std::string const* file_name =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  std::stringstream ss;

  ss << "Is not a file";
  if (file_name != NULL)
  {
    ss << ": \"" << *file_name << "\"";
  }

  return ss.str();
}

std::string
mack::core::files::get_error_message(
    mack::core::files::not_a_directory_error const& error) 
{
  std::string const* file_name =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  std::stringstream ss;

  ss << "Is not a directory";
  if (file_name != NULL)
  {
    ss << ": \"" << *file_name << "\"";
  }

  return ss.str();
}

std::string
mack::core::files::get_error_message(
    mack::core::files::file_read_error const& error) 
{
  std::string const* file_name =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  std::stringstream ss;

  ss << "Error reading file";
  if (file_name != NULL)
  {
    ss << ": \"" << *file_name << "\"";
  }

  return ss.str();
}

std::string
mack::core::files::get_error_message(
    mack::core::files::file_write_error const& error) 
{
  std::string const* file_name =
    boost::get_error_info<mack::core::files::errinfo_file>(error);

  std::stringstream ss;

  ss << "Error writing to file";
  if (file_name != NULL)
  {
    ss << ": \"" << *file_name << "\"";
  }

  return ss.str();
}

std::string
mack::core::files::get_error_message(
    mack::core::files::parse_error const& error) 
{
  std::string const* file_name =
    boost::get_error_info<mack::core::files::errinfo_file>(error);
  size_t const* file_line =
    boost::get_error_info<mack::core::files::errinfo_parse_line>(error);
  std::string const* cause =
    boost::get_error_info<mack::core::files::errinfo_parse_cause>(error);

  std::stringstream ss;

  ss << "Error parsing file";
  if (file_name != NULL)
  {
    ss << " \"" << *file_name << "\"";
  }

  if (file_line != NULL)
  {
    ss << " in line " << *file_line;
  }

  if (cause != NULL)
  {
    ss << ": " << cause;
  }

  return ss.str();
}

