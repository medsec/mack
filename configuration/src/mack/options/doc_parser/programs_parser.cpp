#include "programs_parser.hpp"

#include <mack/options/doc_parser/program_data_parser.hpp>

mack::options::doc_parser::programs_parser::programs_parser(
    boost::filesystem::path const& xml_directory)
  : _xml_dir(xml_directory),
    _children()
{
}

mack::options::doc_parser::programs_parser::~programs_parser()
{
}

void
mack::options::doc_parser::programs_parser::clear()
{
  _children.clear();
}

std::vector<mack::options::doc_parser::program_data> const&
mack::options::doc_parser::programs_parser::get_programs() const
{
  return _children;
}

void
mack::options::doc_parser::programs_parser::start_element(
    std::string const& name,
    mack::core::attributes_type const& attributes)
{
  if (name == "innergroup")
  {
    program_data_parser parser;
    const std::string group_id = attributes.find("refid")->second;
    std::cout << "  Parsing program data in "
      << (_xml_dir / (group_id + ".xml")).native() << std::endl;
    parser.parse_file(_xml_dir / (group_id + ".xml"));
    _children.push_back(parser.get_program_data());
  }
}

void
mack::options::doc_parser::programs_parser::end_element(
    std::string const& name)
{
}

void
mack::options::doc_parser::programs_parser::characters(
    std::string const& text)
{
}

