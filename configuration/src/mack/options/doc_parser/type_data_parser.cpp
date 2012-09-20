#include "type_data_parser.hpp"

#include <mack/options/doc_parser/class_data_parser.hpp>
#include <mack/options/doc_parser/location_parser.hpp>
#include <boost/throw_exception.hpp>

mack::options::doc_parser::type_data_parser::type_data_parser(
    boost::filesystem::path const& src_directory,
    boost::filesystem::path const& xml_directory)
  : _src_dir(boost::filesystem::absolute(src_directory)),
    _xml_dir(xml_directory),
    _data(),
    _children(),
    _get_name(false),
    _current_refid(),
    _refids()
{
}

mack::options::doc_parser::type_data_parser::~type_data_parser()
{
}

void
mack::options::doc_parser::type_data_parser::clear()
{
  _data = type_data();
  _children.clear();
  _get_name = false;
  _current_refid.erase();
  _refids.clear();
}

mack::options::doc_parser::type_data const&
mack::options::doc_parser::type_data_parser::get_type_data() const
{
  return _data;
}

std::vector<mack::options::doc_parser::type_data> const&
mack::options::doc_parser::type_data_parser::get_child_type_datas() const
{
  return _children;
}

void
mack::options::doc_parser::type_data_parser::start_element(
    std::string const& name,
    mack::core::attributes_type const& attributes)
{
  if (name == "compoundname")
  {
    _get_name = true;
  }
  else if (name == "innerclass")
  {
    class_data_parser parser(_src_dir, _xml_dir);
    const std::string class_id = attributes.find("refid")->second;
    parser.set_target_class_id(class_id);
    std::cout << "    Parsing class data in "
      << (_xml_dir / (class_id + ".xml")).native() << std::endl;
    parser.parse_file(_xml_dir / (class_id + ".xml"));
    _data.add_class_data(parser.get_class_data());
  }
  else if (name == "innergroup")
  {
    type_data_parser parser(_src_dir, _xml_dir);
    const std::string group_id = attributes.find("refid")->second;
    std::cout << "    Parsing type data in "
      << (_xml_dir / (group_id + ".xml")).native() << std::endl;
    parser.parse_file(_xml_dir / (group_id + ".xml"));
    const type_data child_type_data = parser.get_type_data();
    for (std::vector<class_data>::const_iterator it =
          child_type_data.get_classes().begin();
        it != child_type_data.get_classes().end();
        ++it)
    {
      _data.add_class_data(*it);
    }

    _children.push_back(child_type_data);
    for (std::vector<type_data>::const_iterator it =
          parser.get_child_type_datas().begin();
        it != parser.get_child_type_datas().end();
        ++it)
    {
      _children.push_back(*it);
    }
  }
  else if (name == "ref")
  {
    _current_refid = attributes.find("refid")->second;
  }
  else if (name == "options_type_class")
  {
    const std::string class_name = attributes.find("value")->second;
    _data.set_type_class_name(class_name);

    const std::string class_id = _refids.find(class_name)->second;
    if (class_id.empty())
    {
      if (_data.get_name().empty())
      {
        BOOST_THROW_EXCEPTION(not_a_class_error()
           << errinfo_class_name(class_name));
      }
      else
      {
        BOOST_THROW_EXCEPTION(not_a_class_error()
           << errinfo_class_name(class_name)
           << errinfo_type_name(_data.get_name()));
      }
    }
    location_parser ref_parser;
    ref_parser.set_target_class_id(class_id);
    std::cout << "    Parsing include path in "
      << (_xml_dir / (class_id + ".xml")).native() << std::endl;
    ref_parser.parse_file(_xml_dir / (class_id + ".xml"));
    const std::string include_file = ref_parser.get_location();
    const std::string include =
      include_file.substr(_src_dir.native().size() + 1);
    _data.set_include_path(include);
  }
}

void
mack::options::doc_parser::type_data_parser::end_element(
    std::string const& name)
{
}

void
mack::options::doc_parser::type_data_parser::characters(
    std::string const& text)
{
  if (_get_name)
  {
    _data.set_name(text);
    _get_name = false;
  }
  else if (!_current_refid.empty())
  {
    _refids.insert(std::pair<std::string, std::string>(text, _current_refid));
    _current_refid.erase();
  }
}

