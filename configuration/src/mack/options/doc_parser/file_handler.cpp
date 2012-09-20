#include "file_handler.hpp"

#include <mack/core/files.hpp>

mack::options::doc_parser::file_handler::file_handler(
    boost::filesystem::path const& src_directory,
    boost::filesystem::path const& xml_directory)
  : _programs_file_path(src_directory / "mack" / "options" / "programs.cpp"),
    _types_directory_path(src_directory / "mack" / "options" / "types"),
    _parser(src_directory, xml_directory),
    _old_type_files()
{
  boost::filesystem::directory_iterator types_dir_it(_types_directory_path);
  boost::filesystem::directory_iterator end_dir_it;
  while (types_dir_it != end_dir_it)
  {
    boost::filesystem::path const& entry = types_dir_it->path();
    if (boost::filesystem::is_regular_file(entry))
    {
      _old_type_files.insert((boost::filesystem::absolute(entry)).native());
    }
    ++types_dir_it;
  }
}

mack::options::doc_parser::file_handler::~file_handler()
{
}

void
mack::options::doc_parser::file_handler::run()
{
  update_programs_file();
  update_type_files();
}

void
mack::options::doc_parser::file_handler::update_programs_file() const
{
  update_file(_programs_file_path, _parser.create_programs_content());
}

void
mack::options::doc_parser::file_handler::update_type_files()
{
  const std::vector<type_files_content> type_file_contents =
    _parser.create_type_files_contents();
  for (std::vector<type_files_content>::const_iterator types_it =
        type_file_contents.begin();
      types_it != type_file_contents.end();
      ++types_it)
  {
    // header
    check_update_file(
        types_it->get_header_file_name(),
        types_it->get_header_content());
    // source
    check_update_file(
        types_it->get_source_file_name(),
        types_it->get_source_content());
  }
  delete_old_files();
}

void
mack::options::doc_parser::file_handler::delete_old_files()
{
  for (std::set<std::string>::const_iterator type_files_it =
        _old_type_files.begin();
      type_files_it != _old_type_files.end();
      ++type_files_it)
  {
    boost::filesystem::remove(boost::filesystem::path(*type_files_it));
  }
  _old_type_files.clear();
}

void
mack::options::doc_parser::file_handler::check_update_file(
    std::string const& file_name,
    std::string const& content)
{
  const boost::filesystem::path file_path =
    boost::filesystem::absolute(_types_directory_path / file_name);

  const std::set<std::string>::iterator found =
    _old_type_files.find(file_path.native());
  if (found == _old_type_files.end())
  {
    // new file
    mack::core::files::write_file(content, file_path);
  }
  else
  {
    // already exists
    update_file(file_path, content);
    _old_type_files.erase(found);
  }
}

void
mack::options::doc_parser::file_handler::update_file(
    boost::filesystem::path const& file_path,
    std::string const& content) const
{
  if (!boost::filesystem::exists(file_path))
  {
    mack::core::files::write_file(content, file_path);
  }
  else if (!mack::core::files::is_content_of(file_path, content))
  {
    boost::filesystem::remove(file_path);
    mack::core::files::write_file(content, file_path);
  }
}

