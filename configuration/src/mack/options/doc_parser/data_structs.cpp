#include "data_structs.hpp"

#include <boost/throw_exception.hpp>
#include <boost/algorithm/string/trim.hpp>

// option data

mack::options::doc_parser::option_data::option_data()
  : _description(),
    _short_flag(),
    _long_flag(),
    _has_default_value(false),
    _default_value(),
    _is_option_switch(false),
    _is_type_option(false),
    _type(),
    _is_for_template_parameter(false),
    _template_parameter()
{
}

mack::options::doc_parser::option_data::~option_data()
{
}

std::string const&
mack::options::doc_parser::option_data::get_description() const
{
  return _description;
}

std::string const&
mack::options::doc_parser::option_data::get_short_flag() const
{
  return _short_flag;
}

std::string const&
mack::options::doc_parser::option_data::get_long_flag() const
{
  return _long_flag;
}

bool
mack::options::doc_parser::option_data::has_default_value() const
{
  return _has_default_value;
}

std::string const&
mack::options::doc_parser::option_data::get_default_value() const
{
  if (!has_default_value())
  {
    BOOST_THROW_EXCEPTION(no_such_value_error());
  }
  return _default_value;
}

bool
mack::options::doc_parser::option_data::is_option_switch() const
{
  return _is_option_switch;
}

bool
mack::options::doc_parser::option_data::is_type_option() const
{
  return _is_type_option;
}

std::string const&
mack::options::doc_parser::option_data::get_type() const
{
  if (!is_type_option())
  {
    BOOST_THROW_EXCEPTION(no_such_value_error());
  }
  return _type;
}

bool
mack::options::doc_parser::option_data::is_for_template_parameter() const
{
  return _is_for_template_parameter;
}

std::string const&
mack::options::doc_parser::option_data::get_corresponding_template_parameter()
  const
{
  if (!is_for_template_parameter())
  {
    BOOST_THROW_EXCEPTION(no_such_value_error());
  }
  return _template_parameter;
}

void
mack::options::doc_parser::option_data::append_description(
    std::string const& description)
{
  _description.append(description);
}

void
mack::options::doc_parser::option_data::set_short_flag(
    std::string const& short_flag)
{
  _short_flag = short_flag;
}

void
mack::options::doc_parser::option_data::set_long_flag(
    std::string const& long_flag)
{
  _long_flag = long_flag;
}

void
mack::options::doc_parser::option_data::set_default_value(
    std::string const& default_value)
{
  if (is_option_switch())
  {
    BOOST_THROW_EXCEPTION(illegal_state_error());
  }
  _default_value = default_value;
  _has_default_value = true;
}

void
mack::options::doc_parser::option_data::set_as_option_switch()
{
  if (has_default_value())
  {
    BOOST_THROW_EXCEPTION(illegal_state_error());
  }
  if (is_type_option())
  {
    BOOST_THROW_EXCEPTION(illegal_state_error());
  }
  if (is_for_template_parameter())
  {
    BOOST_THROW_EXCEPTION(illegal_state_error());
  }
  _is_option_switch = true;
}

void
mack::options::doc_parser::option_data::set_type(
    std::string const& type)
{
  if (is_option_switch())
  {
    BOOST_THROW_EXCEPTION(illegal_state_error());
  }
  _type = type;
  _is_type_option = true;
}

void
mack::options::doc_parser::option_data::set_corresponding_template_parameter(
    std::string const& template_parameter)
{
  if (is_option_switch())
  {
    BOOST_THROW_EXCEPTION(illegal_state_error());
  }
  _template_parameter = template_parameter;
  _is_for_template_parameter = true;
}

void
mack::options::doc_parser::option_data::trim_description()
{
  boost::algorithm::trim(_description);
}

// class data

mack::options::doc_parser::class_data::class_data()
  : _class_name(),
    _include(),
    _brief(),
    _details(),
    _extends(),
    _options(),
    _template_parameters()
{
}

mack::options::doc_parser::class_data::~class_data()
{
}

void
mack::options::doc_parser::class_data::set_class_name(
    std::string const& class_name)
{
  _class_name = class_name;
}

void
mack::options::doc_parser::class_data::set_include_path(
    std::string const& include_path)
{
  _include = include_path;
}

void
mack::options::doc_parser::class_data::add_to_brief_description(
    std::string const& brief_description)
{
  _brief.append(brief_description);
}

void
mack::options::doc_parser::class_data::add_to_detailed_description(
    std::string const& detailed_description)
{
  _details.append(detailed_description);
}

void
mack::options::doc_parser::class_data::add_extended_class(
    std::string const& extended_class)
{
  _extends.push_back(extended_class);
}

void
mack::options::doc_parser::class_data::add_option(
    mack::options::doc_parser::option_data const& option)
{
  _options.push_back(option);
}

void
mack::options::doc_parser::class_data::add_template_parameter(
    std::string const& template_parameter)
{
  _template_parameters.push_back(template_parameter);
}

void
mack::options::doc_parser::class_data::trim_description()
{
  boost::algorithm::trim(_details);
}

std::string const&
mack::options::doc_parser::class_data::get_class_name() const
{
  return _class_name;
}

std::string const&
mack::options::doc_parser::class_data::get_include_path() const
{
  return _include;
}

std::string const&
mack::options::doc_parser::class_data::get_brief_description() const
{
  return _brief;
}

std::string const&
mack::options::doc_parser::class_data::get_detailed_description() const
{
  return _details;
}

std::vector<std::string> const&
mack::options::doc_parser::class_data::get_extended_classes() const
{
  return _extends;
}

std::vector<mack::options::doc_parser::option_data> const&
mack::options::doc_parser::class_data::get_options() const
{
  return _options;
}

mack::options::doc_parser::option_data const&
mack::options::doc_parser::class_data::get_option_for_template_parameter(
    std::string const& parameter) const
{
  for (std::vector<option_data>::const_iterator it = _options.begin();
      it != _options.end();
      ++it)
  {
    if (it->is_for_template_parameter()
        && it->get_corresponding_template_parameter() == parameter)
    {
      return *it;
    }
  }
  BOOST_THROW_EXCEPTION(no_option_for_tparam_error()
      << errinfo_template_parameter(parameter));
}

std::vector<std::string> const&
mack::options::doc_parser::class_data::get_template_parameters() const
{
  return _template_parameters;
}

// type_data

mack::options::doc_parser::type_data::type_data()
  : _name(),
    _has_type_class(false),
    _type_class_name(),
    _include(),
    _classes()
{
}

mack::options::doc_parser::type_data::~type_data()
{
}

std::string const&
mack::options::doc_parser::type_data::get_name() const
{
  return _name;
}

bool
mack::options::doc_parser::type_data::has_type_class() const
{
  return _has_type_class;
}

std::string const&
mack::options::doc_parser::type_data::get_type_class_name() const
{
  if (!has_type_class())
  {
    BOOST_THROW_EXCEPTION(no_such_value_error());
  }
  return _type_class_name;
}

std::string const&
mack::options::doc_parser::type_data::get_include_path() const
{
  if (!has_type_class())
  {
    BOOST_THROW_EXCEPTION(no_such_value_error());
  }
  return _include;
}

std::vector<mack::options::doc_parser::class_data> const&
mack::options::doc_parser::type_data::get_classes() const
{
  return _classes;
}

void
mack::options::doc_parser::type_data::set_name(
    std::string const& name)
{
  _name = name;
}

void
mack::options::doc_parser::type_data::set_type_class_name(
    std::string const& type_class_name)
{
  _type_class_name = type_class_name;
  _has_type_class = true;
}

void
mack::options::doc_parser::type_data::set_include_path(
    std::string const& include_path)
{
  _include = include_path;
  _has_type_class = true;
}

void
mack::options::doc_parser::type_data::add_class_data(
    mack::options::doc_parser::class_data const& data)
{
  _classes.push_back(data);
}

// program_data

mack::options::doc_parser::program_data::program_data()
  : _name(),
    _brief(),
    _details(),
    _options()
{
}

mack::options::doc_parser::program_data::~program_data()
{
}

std::string const&
mack::options::doc_parser::program_data::get_name() const
{
  return _name;
}

std::string const&
mack::options::doc_parser::program_data::get_brief_description() const
{
  return _brief;
}

std::string const&
mack::options::doc_parser::program_data::get_detailed_description() const
{
  return _details;
}

std::vector<mack::options::doc_parser::option_data> const&
mack::options::doc_parser::program_data::get_options() const
{
  return _options;
}

void
mack::options::doc_parser::program_data::set_name(
    std::string const& name)
{
  _name = name;
}

void
mack::options::doc_parser::program_data::add_to_brief_description(
    std::string const& brief_description)
{
  _brief.append(brief_description);
}

void
mack::options::doc_parser::program_data::add_to_detailed_description(
    std::string const& detailed_description)
{
  _details.append(detailed_description);
}

void
mack::options::doc_parser::program_data::add_option(
    mack::options::doc_parser::option_data const& option)
{
  _options.push_back(option);
}

// operators

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::option_data const& data)
{
  output_stream << "{ OPTION" << std::endl;
  output_stream << "short flag: \""
    << data.get_short_flag() << "\"" << std::endl;
  output_stream << "long flag: \""
    << data.get_long_flag() << "\"" << std::endl;
  output_stream << "description: \""
    << data.get_description() << "\"" << std::endl;
  output_stream << "is switch: \""
    << data.is_option_switch() << "\"" << std::endl;
  if (data.has_default_value())
  {
    output_stream << "default: \""
      << data.get_default_value() << "\"" << std::endl;
  }
  if (data.is_type_option())
  {
    output_stream << "type: \""
      << data.get_type() << "\"" << std::endl;
  }
  if (data.is_for_template_parameter())
  {
    output_stream << "tparam: \""
      << data.get_corresponding_template_parameter() << "\"" << std::endl;
  }
  output_stream << "}" << std::endl;
  return output_stream;
}

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::class_data const& data)
{
  output_stream << "{ CLASS" << std::endl;
  output_stream << "class name: \""
    << data.get_class_name() << "\"" << std::endl;
  output_stream << "include path: \""
    << data.get_include_path() << "\"" << std::endl;
  output_stream << "brief description: \""
    << data.get_brief_description() << "\"" << std::endl;
  output_stream << "detailed description: \""
    << data.get_detailed_description() << "\"" << std::endl;
  for (std::vector<std::string>::const_iterator it
        = data.get_extended_classes().begin();
      it != data.get_extended_classes().end();
      ++it)
  {
    output_stream << "extends: \""
      << *it << "\"" << std::endl;
  }
  for (std::vector<mack::options::doc_parser::option_data>::const_iterator it
        = data.get_options().begin();
      it != data.get_options().end();
      ++it)
  {
    output_stream << *it;
  }
  for (std::vector<std::string>::const_iterator it
        = data.get_template_parameters().begin();
      it != data.get_template_parameters().end();
      ++it)
  {
    output_stream << "tparam: \""
      << *it << "\"" << std::endl;
  }
  output_stream << "}" << std::endl;
  return output_stream;
}

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::type_data const& data)
{
  output_stream << "{ TYPE" << std::endl;
  output_stream << "name: \""
    << data.get_name() << "\"" << std::endl;
  if (data.has_type_class())
  {
    output_stream << "type class name: \""
      << data.get_type_class_name() << "\"" << std::endl;
    output_stream << "type include path: \""
      << data.get_include_path() << "\"" << std::endl;
  }
  for (std::vector<mack::options::doc_parser::class_data>::const_iterator it
        = data.get_classes().begin();
      it != data.get_classes().end();
      ++it)
  {
    output_stream << *it;
  }
  output_stream << "}" << std::endl;
  return output_stream;
}

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::program_data const& data)
{
  output_stream << "{ PROGRAM" << std::endl;
  output_stream << "name: \""
    << data.get_name() << "\"" << std::endl;
  for (std::vector<mack::options::doc_parser::option_data>::const_iterator it
        = data.get_options().begin();
      it != data.get_options().end();
      ++it)
  {
    output_stream << *it;
  }
  output_stream << "}" << std::endl;
  return output_stream;
}

