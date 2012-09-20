#include "program_data_parser.hpp"

mack::options::doc_parser::program_data_parser::program_data_parser()
  : _data(),
    _option(),
    _get_name(false),
    _get_brief(false),
    _get_details(false),
    _is_simple_sect(false),
    _get_option_text(false)
{
}

mack::options::doc_parser::program_data_parser::~program_data_parser()
{
}

void
mack::options::doc_parser::program_data_parser::clear()
{
  _get_name = false;
  _get_brief = false;
  _get_details = false;
  _is_simple_sect = false;
  _get_option_text = false;
}

mack::options::doc_parser::program_data const&
mack::options::doc_parser::program_data_parser::get_program_data() const
{
  return _data;
}

void
mack::options::doc_parser::program_data_parser::start_element(
    std::string const& name,
    mack::core::attributes_type const& attributes)
{
  if (name == "compoundname")
  {
    _get_name = true;
  }
  else if (name == "briefdescription")
  {
    _get_brief = true;
  }
  else if (name == "detaileddescription")
  {
    _get_details = true;
  }
  else if (_get_details && name == "simplesect")
  {
    _is_simple_sect = true;
  }
  else if (name == "option")
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

void
mack::options::doc_parser::program_data_parser::end_element(
    std::string const& name)
{
  if (_get_option_text && name == "simplesect")
  {
    _get_option_text = false;
    _option.trim_description();
    _data.add_option(_option);
    _option = option_data();
    _is_simple_sect = false;
  }
  else if (name == "briefdescription")
  {
    _get_brief = false;
  }
  else if (name == "detaileddescription")
  {
    _get_details = false;
  }
  else if (_get_details && name == "simplesect")
  {
    _is_simple_sect = false;
  }
}

void
mack::options::doc_parser::program_data_parser::characters(
    std::string const& text)
{
  if (_get_name)
  {
    _data.set_name(text);
    _get_name = false;
  }
  else if (_get_brief)
  {
    _data.add_to_brief_description(text);
  }
  else if (_get_details && !_is_simple_sect)
  {
    _data.add_to_detailed_description(text);
  }
  else if (_get_option_text)
  {
    _option.append_description(text);
  }
}

