#include "class_data_parser.hpp"

mack::options::doc_parser::class_data_parser::class_data_parser(
    boost::filesystem::path const& src_directory,
    boost::filesystem::path const& xml_directory)
  : _src_dir(boost::filesystem::absolute(src_directory).native()),
    _xml_dir(xml_directory),
    _target_class_id(),
    _data(),
    _option(),
    _stack(),
    _get_name(false),
    _get_extends(false),
    _get_tparam(false),
    _get_brief(false),
    _is_simple_sect(false),
    _get_detailed(false),
    _get_option_text(false)
{
}

mack::options::doc_parser::class_data_parser::~class_data_parser()
{
}

void
mack::options::doc_parser::class_data_parser::clear()
{
  _data = class_data();
  _option = option_data();
  _stack.clear();
  _get_name = false;
  _get_extends = false;
  _get_tparam = false;
  _get_brief = false;
  _is_simple_sect = false;
  _get_detailed = false;
  _get_option_text = false;
}

void
mack::options::doc_parser::class_data_parser::set_target_class_id(
    std::string const& target_class_id)
{
  _target_class_id = target_class_id;
}

mack::options::doc_parser::class_data const&
mack::options::doc_parser::class_data_parser::get_class_data() const
{
  return _data;
}

void
mack::options::doc_parser::class_data_parser::start_element(
    std::string const& name,
    mack::core::attributes_type const& attributes)
{
  if (_stack.empty())
  {
    if (name == "compounddef")
    {
      if (attributes.find("id")->second == _target_class_id)
      {
        _stack.push_back(name);
      }
    }
  }
  else
  {
    if (_stack.size() == 1)
    {
      if (name == "compoundname")
      {
        _get_name = true;
      }
      else if (name == "basecompoundref")
      {
        _get_extends = true;
      }
      else if (name == "location")
      {
        _data.set_include_path(
            attributes.find("file")->second.substr(
              _src_dir.size() + 1));
      }
    }
    else if (_stack.size() == 2)
    {
      if (name == "para")
      {
        if (_stack[1] == "briefdescription")
        {
          _get_brief = true;
        }
        else if (_stack[1] == "detaileddescription")
        {
          _get_detailed = true;
        }
      }
    }
    else if (_stack.size() == 3)
    {
      if (name == "simplesect"
          && _stack[2] == "para"
          && _stack[1] == "detaileddescription")
      {
        _is_simple_sect = true;
      }
      else if (name == "type"
          && _stack[2] == "param"
          && _stack[1] == "templateparamlist")
      {
        _get_tparam = true;
      }
    }
    else if (_stack.size() == 5)
    {
      if (name == "option"
          && _stack[4] == "para"
          && _stack[3] == "simplesect"
          && _stack[2] == "para"
          && _stack[1] == "detaileddescription")
      {
        _option.set_short_flag(attributes.find("short_flag")->second);
        _option.set_long_flag(attributes.find("flag")->second);
        const std::string type = attributes.find("type")->second;
        if (type == "switch")
        {
          _option.set_as_option_switch();
        }
        else if (type != "value")
        {
          _option.set_type(type.substr(type.find(' ') + 1));
        }

        if (attributes.count("default") == 1)
        {
          _option.set_default_value(attributes.find("default")->second);
        }

        if (attributes.count("tparam") == 1)
        {
          _option.set_corresponding_template_parameter(
              attributes.find("tparam")->second);
        }

        _get_option_text = true;
      }
    }
    _stack.push_back(name);
  }
}

void
mack::options::doc_parser::class_data_parser::end_element(
    std::string const& name)
{
  if (!_stack.empty())
  {
    if (_get_brief && _stack.size() == 3)
    {
      _get_brief = false;
    }
    else if (_get_detailed && _stack.size() == 3)
    {
      _get_detailed = false;
    }
    else if (_is_simple_sect && _stack.size() == 4)
    {
      _is_simple_sect = false;
    }
    else if (_get_option_text && _stack.size() == 5)
    {
      _get_option_text = false;
      _option.trim_description();
      _data.add_option(_option);
      _option = option_data();
    }
    _stack.pop_back();
  }
  else if (name == "doxygen")
  {
    _data.trim_description();
  }
}

void
mack::options::doc_parser::class_data_parser::characters(
    std::string const& text)
{
  if (_get_name)
  {
    _data.set_class_name(text);
    _get_name = false;
  }
  else if (_get_extends)
  {
    _data.add_extended_class(text);
    _get_extends = false;
  }
  else if (_get_tparam)
  {
    _data.add_template_parameter(text.substr(text.find(' ') + 1));
    _get_tparam = false;
  }
  else if (_get_brief)
  {
    _data.add_to_brief_description(text);
  }
  else if (_get_detailed && !_is_simple_sect)
  {
    _data.add_to_detailed_description(text);
  }
  else if (_get_option_text)
  {
    _option.append_description(text);
  }
}

