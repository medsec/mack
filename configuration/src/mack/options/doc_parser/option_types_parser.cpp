#include "option_types_parser.hpp"

#include <mack/options/doc_parser/type_data_parser.hpp>

mack::options::doc_parser::option_types_parser::option_types_parser(
    boost::filesystem::path const& src_directory,
    boost::filesystem::path const& xml_directory)
  : _src_dir(boost::filesystem::absolute(src_directory)),
    _xml_dir(xml_directory),
    _children()
{
}

mack::options::doc_parser::option_types_parser::~option_types_parser()
{
}

void
mack::options::doc_parser::option_types_parser::clear()
{
  _children.clear();
}

std::vector<mack::options::doc_parser::type_data> const&
mack::options::doc_parser::option_types_parser::get_child_type_datas() const
{
  return _children;
}

void
mack::options::doc_parser::option_types_parser::start_element(
    std::string const& name,
    mack::core::attributes_type const& attributes)
{
  if (name == "innergroup")
  {
    type_data_parser parser(_src_dir, _xml_dir);
    const std::string group_id = attributes.find("refid")->second;
    std::cout << "  Parsing type data in "
      << (_xml_dir / (group_id + ".xml")).native() << std::endl;
    parser.parse_file(_xml_dir / (group_id + ".xml"));
    const type_data child_type_data = parser.get_type_data();
    _children.push_back(child_type_data);

    for (std::vector<type_data>::const_iterator it =
          parser.get_child_type_datas().begin();
        it != parser.get_child_type_datas().end();
        ++it)
    {
      _children.push_back(*it);
    }
  }
}

void
mack::options::doc_parser::option_types_parser::end_element(
    std::string const& name)
{
}

void
mack::options::doc_parser::option_types_parser::characters(
    std::string const& text)
{
}

