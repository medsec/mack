#include "parser.hpp"

#include <mack/core/files.hpp>

#include <mack/options/doc_parser/option_types_parser.hpp>
#include <mack/options/doc_parser/programs_parser.hpp>

#include <boost/algorithm/string.hpp>

#include <sstream>

std::string
replace_line_endings(
    std::string const& text)
{
  std::string copy(text);
  size_t index = copy.find('\n');
  while (index != std::string::npos)
  {
    copy = copy.replace(index, 1, "\\n");
    index = copy.find('\n', index);
  }
  return copy;
}

mack::options::doc_parser::type_files_content::type_files_content()
  : _type_name(),
    _header_file(),
    _source_file(),
    _contains_cuda_includes()
{
}

mack::options::doc_parser::type_files_content::type_files_content(
      std::string const& type_name,
      std::string const& header_file,
      std::string const& source_file,
      const bool contains_cuda_includes)
  : _type_name(type_name),
    _header_file(header_file),
    _source_file(source_file),
    _contains_cuda_includes(contains_cuda_includes)
{
}

mack::options::doc_parser::type_files_content::~type_files_content()
{
}

std::string const&
mack::options::doc_parser::type_files_content::get_type_name() const
{
  return _type_name;
}

std::string const&
mack::options::doc_parser::type_files_content::get_header_content() const
{
  return _header_file;
}

std::string const&
mack::options::doc_parser::type_files_content::get_source_content() const
{
  return _source_file;
}

std::string
mack::options::doc_parser::type_files_content::get_header_file_name() const
{
  return get_type_name() + ".hpp";
}

std::string
mack::options::doc_parser::type_files_content::get_source_file_name() const
{
  if (_contains_cuda_includes)
  {
    return get_type_name() + ".cu";
  }
  else
  {
    return get_type_name() + ".cpp";
  }
}

mack::options::doc_parser::parser::parser()
  : _types(),
    _programs()
{
}

mack::options::doc_parser::parser::parser(
    boost::filesystem::path const& src_directory,
    boost::filesystem::path const& xml_directory)
  : _types(),
    _programs()
{
  option_types_parser types_parser(src_directory, xml_directory);

  std::cout << "Parsing option types in "
    << (xml_directory / "group__option__types.xml").native() << std::endl;
  types_parser.parse_file(xml_directory / "group__option__types.xml");
  const std::vector<type_data> types = types_parser.get_child_type_datas();

  for (std::vector<type_data>::const_iterator types_it = types.begin();
      types_it != types.end();
      ++types_it)
  {
    _types.insert(std::pair<std::string, type_data>(
          types_it->get_name(),
          *types_it));
  }

  programs_parser programs_parser(xml_directory);
  std::cout << "Parsing programs in "
    << (xml_directory / "group__programs__group.xml").native() << std::endl;
  programs_parser.parse_file(xml_directory / "group__programs__group.xml");
  _programs = programs_parser.get_programs();
}

mack::options::doc_parser::parser::~parser()
{
}

std::string
mack::options::doc_parser::parser::create_programs_content() const
{
  std::set<std::string> includes;

  std::stringstream stream;
  const std::string body = create_programs_content_body(includes);
  stream << create_programs_content_includes(includes);
  stream << body;

  return stream.str();
}

std::vector<mack::options::doc_parser::type_files_content>
mack::options::doc_parser::parser::create_type_files_contents() const
{
  std::vector<type_files_content> contents;
  for (std::map<std::string, type_data>::const_iterator type_it = _types.begin();
      type_it != _types.end();
      ++type_it)
  {
    contents.push_back(create_type_files_content(type_it->second));
  }
  return contents;
}

std::string
mack::options::doc_parser::parser::create_programs_content_includes(
    std::set<std::string>& includes) const
{
  std::stringstream stream;
  stream << "#include \"programs.hpp\"" << std::endl;
  stream << std::endl;
  stream << "#include <mack/options/option.hpp>" << std::endl;
  stream << "#include <mack/options/selection_option.hpp>" << std::endl;
  stream << "#include <mack/options/option_switch.hpp>" << std::endl;
  stream << "#include <mack/options/type_option.hpp>" << std::endl;
  stream << "#include <boost/throw_exception.hpp>" << std::endl;
  stream << std::endl;
  for (std::set<std::string>::const_iterator include_it = includes.begin();
      include_it != includes.end();
      ++include_it)
  {
    stream << "#include <" << *include_it << ">" << std::endl;
  }
  stream << std::endl;

  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_programs_content_body(
    std::set<std::string>& includes) const
{
  std::stringstream stream;

  stream << "mack::options::program_options*" << std::endl;
  stream << "mack::options::programs::create_program_option(" << std::endl;
  stream << "    std::string const& program_name)" << std::endl;
  stream << "{" << std::endl;
  stream << "  std::string brief;" << std::endl;
  stream << "  std::string details;" << std::endl;
  stream << "  std::vector<option*> options;" << std::endl;
  stream << std::endl;

  bool first_program = true;
  for (std::vector<program_data>::const_iterator program_it = _programs.begin();
      program_it != _programs.end();
      ++program_it)
  {
    stream << "  ";
    if (first_program)
    {
      first_program = false;
    }
    else
    {
      stream << "else ";
    }
    stream << "if (program_name == \"" << program_it->get_name() << "\")"
      << std::endl;
    stream << "  {" << std::endl;
    stream << create_program_content(*program_it, includes);
    stream << "  }" << std::endl;
  }

  if (!first_program)
  {
    stream << "  else" << std::endl;
  }

  stream << "  {" << std::endl;
  stream << "    BOOST_THROW_EXCEPTION(no_such_program_error()" << std::endl;
  stream << "        << errinfo_program_name(program_name));" << std::endl;
  stream << "  }" << std::endl;
  stream << "  return new program_options(" << std::endl;
  stream << "      program_name," << std::endl;
  stream << "      brief," << std::endl;
  stream << "      details," << std::endl;
  stream << "      options);" << std::endl;
  stream << "}" << std::endl;
  stream << std::endl;

  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_program_content(
    program_data const& data,
    std::set<std::string>& includes) const
{
  std::vector<option_data> const& options = data.get_options();
  std::stringstream stream;
  stream << "    brief = \""
    << replace_line_endings(data.get_brief_description()) << "\";"<< std::endl;
  stream << "    details = \""
    << replace_line_endings(data.get_detailed_description()) << "\";"
    << std::endl;
  for (std::vector<option_data>::const_iterator option_it = options.begin();
      option_it != options.end();
      ++option_it)
  {
    stream << "    options.push_back(" << std::endl;
    stream << create_option(*option_it, includes);
    stream << "      );" << std::endl;
  }
  return stream.str();
}

mack::options::doc_parser::type_files_content
mack::options::doc_parser::parser::create_type_files_content(
    type_data const& data) const
{
  const std::string header =
    create_type_file_header(data);
  bool contains_cuda_includes;
  const std::string source =
    create_type_file_source(data, contains_cuda_includes);

  return type_files_content(
      data.get_name(), header, source, contains_cuda_includes);
}

std::string
mack::options::doc_parser::parser::create_type_file_header(
    type_data const& data) const
{
  const std::string name = data.get_name();
  const std::string upper_name = boost::to_upper_copy(name);

  std::stringstream stream;
  stream << "#ifndef __MACK_OPTIONS_TYPES_" << upper_name << "_HPP__"
    << std::endl;
  stream << "#define __MACK_OPTIONS_TYPES_" << upper_name << "_HPP__"
    << std::endl;
  stream << std::endl;
  stream << "#include <mack/options/option_type.hpp>" << std::endl;
  stream << "#include <string>" << std::endl;
  stream << "#include <vector>" << std::endl;
  stream << std::endl;
  stream << "namespace mack {" << std::endl;
  stream << "namespace options {" << std::endl;
  stream << "namespace types {" << std::endl;
  stream << std::endl;
  stream << "class " << name << " : public option_type {" << std::endl;
  stream << std::endl;
  stream << "  public:" << std::endl;
  stream << std::endl;
  stream << "    " << name << "();" << std::endl;
  stream << std::endl;
  stream << "    virtual" << std::endl;
  stream << "    ~" << name << "();" << std::endl;
  stream << std::endl;
  stream << "    virtual" << std::endl;
  stream << "    std::vector<std::string> const&" << std::endl;
  stream << "    get_class_names() const;" << std::endl;
  stream << std::endl;
  stream << "    virtual" << std::endl;
  stream << "    options*" << std::endl;
  stream << "    get_options(" << std::endl;
  stream << "        std::string const& class_name) const;" << std::endl;
  stream << std::endl;
  stream << "    virtual" << std::endl;
  stream << "    value*" << std::endl;
  stream << "    create_value(" << std::endl;
  stream << "        std::string const& class_name," << std::endl;
  stream << "        values const* values," << std::endl;
  stream << "        const bool is_template_value) const;" << std::endl;
  stream << std::endl;
  stream << "  private:" << std::endl;
  stream << std::endl;
  stream << "    std::vector<std::string> _class_names;" << std::endl;
  stream << std::endl;
  stream << "}; // class " << name << std::endl;
  stream << std::endl;
  stream << "} // namespace types" << std::endl;
  stream << "} // namespace options" << std::endl;
  stream << "} // namespace mack" << std::endl;
  stream << std::endl;
  stream << "#endif /* __MACK_OPTIONS_TYPES_" << upper_name << "_HPP__ */"
    << std::endl;
  stream << std::endl;
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source(
    type_data const& data,
    bool& contains_cuda_includes) const
{
  contains_cuda_includes = false;

  std::stringstream stream;
  std::set<std::string> includes;

  const std::string body = create_type_file_source_body(data, includes);
  stream << create_type_file_source_includes(data, includes);
  stream << body;

  for (std::set<std::string>::const_iterator include_it = includes.begin();
      include_it != includes.end();
      ++include_it)
  {
    if (include_it->substr(include_it->size() - 3) == ".cu"
        || include_it->substr(include_it->size() - 4) == ".cuh")
    {
      contains_cuda_includes = true;
      break;
    }
  }

  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source_includes(
    type_data const& data,
    std::set<std::string>& includes) const
{
  const std::string name = data.get_name();
  std::stringstream stream;
  if (data.has_type_class())
  {
    includes.insert(data.get_include_path());
  }

  stream << "#include \"" << name << ".hpp\"" << std::endl;
  stream << std::endl;

  for (std::set<std::string>::const_iterator it = includes.begin();
      it != includes.end();
      ++it)
  {
    stream << "#include <" << (*it) << ">" << std::endl;
  }
  stream << std::endl;

  stream << "#include <mack/options/exceptions.hpp>" << std::endl;
  stream << "#include <mack/options/option.hpp>" << std::endl;
  stream << "#include <mack/options/selection_option.hpp>" << std::endl;
  stream << "#include <mack/options/type_option.hpp>" << std::endl;
  stream << "#include <mack/options/option_switch.hpp>" << std::endl;
  stream << "#include <boost/throw_exception.hpp>" << std::endl;
  stream << std::endl;

  return stream.str();;
}

std::string
mack::options::doc_parser::parser::create_type_file_source_body(
    type_data const& data,
    std::set<std::string>& includes) const
{
  std::stringstream stream;
  stream << create_type_file_source_constructor(data, includes);
  stream << create_type_file_source_destructor(data, includes);
  stream << create_type_file_source_get_class_names(data, includes);
  stream << create_type_file_source_get_options(data, includes);
  stream << create_type_file_source_create_value(data, includes);
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source_constructor(
    type_data const& data,
    std::set<std::string>& includes) const
{
  const std::string name = data.get_name();
  std::vector<class_data> const& classes = data.get_classes();
  std::stringstream stream;

  stream << "mack::options::types::" << name << "::" << name <<"()" << std::endl;
  stream << "  : _class_names()" << std::endl;
  stream << "{" << std::endl;
  for (std::vector<class_data>::const_iterator it = classes.begin();
      it != classes.end();
      ++it)
  {
    stream << "  _class_names.push_back(\"" << it->get_class_name() << "\");"
      << std::endl;
  }
  stream << "}" << std::endl;
  stream << std::endl;
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source_destructor(
    type_data const& data,
    std::set<std::string>& includes) const
{
  const std::string name = data.get_name();
  std::stringstream stream;

  stream << "mack::options::types::" << name <<"::~" << name << "()"
    << std::endl;
  stream << "{" << std::endl;
  stream << "}" << std::endl;
  stream << std::endl;
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source_get_class_names(
    type_data const& data,
    std::set<std::string>& includes) const
{
  const std::string name = data.get_name();
  std::stringstream stream;

  stream << "std::vector<std::string> const&" << std::endl;
  stream << "mack::options::types::" << name << "::get_class_names() const"
    << std::endl;
  stream << "{" << std::endl;
  stream << "  return _class_names;" << std::endl;
  stream << "}" << std::endl;
  stream << std::endl;
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source_get_options(
    type_data const& data,
    std::set<std::string>& includes) const
{
  const std::string name = data.get_name();
  std::vector<class_data> const& classes = data.get_classes();
  std::stringstream stream;

  bool first_option = true;
  stream << "mack::options::options*" << std::endl;
  stream << "mack::options::types::" << name << "::get_options(" << std::endl;
  stream << "    std::string const& class_name) const" << std::endl;
  stream << "{" << std::endl;
  stream << "  std::vector<option*> opts;" << std::endl;
  stream << "  std::string brief;" << std::endl;
  stream << "  std::string details;" << std::endl;
  for (std::vector<class_data>::const_iterator it = classes.begin();
      it != classes.end();
      ++it)
  {
    stream << "  ";
    if (first_option)
    {
      first_option = false;
    }
    else
    {
      stream << "else ";
    }
    stream << "if (class_name == \"" << it->get_class_name() << "\")"
      << std::endl;
    stream << "  {" << std::endl;
    stream << "    brief = \""
      << replace_line_endings(it->get_brief_description()) << "\";"
      << std::endl;
    stream << "    details = \""
      << replace_line_endings(it->get_detailed_description()) << "\";"
      << std::endl;
    for (std::vector<option_data>::const_iterator oit =
          it->get_options().begin();
        oit != it->get_options().end();
        ++oit)
    {
      stream << "    {" << std::endl;
      stream << "      opts.push_back(" << std::endl;
      stream << create_option(*oit, includes);
      stream << "            );" << std::endl;
      stream << "    }" << std::endl;
    }
    stream << "  }" << std::endl;
  }
  if (!first_option)
  {
    stream << "  else" << std::endl;
  }
  stream << "  {" << std::endl;
  stream << "    BOOST_THROW_EXCEPTION(invalid_value_error()" << std::endl;
  stream << "        << errinfo_option_value(class_name));" << std::endl;
  stream << "  }" << std::endl;
  stream << "  return new options(brief, details, opts);" << std::endl;
  stream << "}" << std::endl;
  stream << std::endl;
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_type_file_source_create_value(
    type_data const& data,
    std::set<std::string>& includes) const
{
  const std::string name = data.get_name();
  std::vector<class_data> const& classes = data.get_classes();
  std::stringstream stream;

  stream << "mack::options::value*" << std::endl;
  stream << "mack::options::types::" << name << "::create_value(" << std::endl;
  stream << "    std::string const& class_name," << std::endl;
  stream << "    mack::options::values const* values," << std::endl;
  stream << "    const bool is_template_value) const" << std::endl;
  stream << "{" << std::endl;
  bool first_class = true;
  for (std::vector<class_data>::const_iterator it = classes.begin();
      it != classes.end();
      ++it)
  {
    includes.insert(it->get_include_path());
    stream << "  ";
    if (first_class)
    {
      first_class = false;
    }
    else
    {
      stream << "else ";
    }
    stream << "if (class_name == \"" << it->get_class_name() << "\")"
      << std::endl;
    stream << "  {" << std::endl;

    if (data.has_type_class())
    {
      std::vector<std::string> const& templates = it->get_template_parameters();
      if (templates.empty())
      {
        stream << "    if (is_template_value)" << std::endl;
        stream << "    {" << std::endl;
        stream << "      return new value(" << std::endl;
        stream << "            new " << it->get_class_name();
        if (it->get_options().empty())
        {
          stream << "," << std::endl;
        }
        else
        {
          stream << "(values)," << std::endl;
        }
        stream << "          \"" << it->get_class_name() << "\");" << std::endl;
        stream << "    }" << std::endl;
        stream << "    else" << std::endl;
        stream << "    {" << std::endl;
        stream << "      return new value(" << std::endl;
        stream << "          (" << data.get_type_class_name() << "*)"
          << std::endl;
        stream << "            new " << it->get_class_name();
        if (it->get_options().empty())
        {
          stream << "," << std::endl;
        }
        else
        {
          stream << "(values)," << std::endl;
        }
        stream << "          \"" << it->get_class_name() << "\");" << std::endl;
        stream << "    }" << std::endl;
      }
      else
      {
        stream << "    if (is_template_value)" << std::endl;
        stream << "    {" << std::endl;
        std::vector<std::string> stack;
        stream << create_template_class(
            it->get_class_name(),
            *it,
            false,
            data.get_type_class_name(),
            templates.begin(),
            templates.end(),
            stack,
            includes);
        stream << "    }" << std::endl;
        stream << "    else" << std::endl;
        stack.clear();
        stream << "    {" << std::endl;
        stream << create_template_class(
            it->get_class_name(),
            *it,
            true,
            data.get_type_class_name(),
            templates.begin(),
            templates.end(),
            stack,
            includes);
        stream << "    }" << std::endl;
      }
    }
    else // has no type class
    {
      std::vector<std::string> const& templates = it->get_template_parameters();
      stream << "    if (is_template_value)" << std::endl;
      stream << "    {" << std::endl;
      if (templates.empty())
      {
        stream << "      return new value(" << std::endl;
        stream << "            new " << it->get_class_name();
        if (it->get_options().empty())
        {
          stream << "," << std::endl;
        }
        else
        {
          stream << "(values)," << std::endl;
        }
        stream << "          \"" << it->get_class_name() << "\");" << std::endl;
      }
      else
      {
        std::vector<std::string> stack;
        stream << create_template_class(
            it->get_class_name(),
            *it,
            false,
            "",
            templates.begin(),
            templates.end(),
            stack,
            includes);
      }
      stream << "    }" << std::endl;
      stream << "    else" << std::endl;
      stream << "    {" << std::endl;
      stream << "    BOOST_THROW_EXCEPTION(invalid_value_error()" << std::endl;
      stream << "        << errinfo_option_value(class_name));" << std::endl;
      stream << "    }" << std::endl;
    }
    stream << "  }" << std::endl;
  }
  if (!first_class)
  {
    stream << "  else" << std::endl;
  }
  stream << "  {" << std::endl;
  stream << "    BOOST_THROW_EXCEPTION(invalid_value_error()" << std::endl;
  stream << "        << errinfo_option_value(class_name));" << std::endl;
  stream << "  }" << std::endl;
  stream << "}" << std::endl;
  stream << std::endl;
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_template_class(
    std::string const& class_name,
    class_data const& data,
    const bool has_type_class,
    std::string const& type_class_name,
    std::vector<std::string>::const_iterator current,
    std::vector<std::string>::const_iterator const& end,
    std::vector<std::string>& stack,
    std::set<std::string>& includes) const
{
  std::stringstream stream;
  std::string indent = std::string();
  indent.append(6 + stack.size() * 2,' ');
  if (current == end)
  {
    stream << indent << "return new value(" << std::endl;
    if (has_type_class)
    {
      stream << "      (" << type_class_name << "*)" << std::endl;
    }
    stream << indent << "      new " << class_name << "<";
    bool first = true;
    for (std::vector<std::string>::const_iterator stack_it = stack.begin();
        stack_it != stack.end();
        ++stack_it)
    {
      if (first)
      {
        first = false;
      }
      else
      {
        stream << ",";
      }
      stream << *stack_it;
    }
    stream << ">(values),"
      << std::endl;
    stream << indent << "    \"" << class_name << "\");" << std::endl;
  }
  else
  {
    const std::string parameter = *current;
    option_data const& option =
      data.get_option_for_template_parameter(parameter);
    if (!option.is_type_option())
    {
      BOOST_THROW_EXCEPTION(illegal_state_error()
          << errinfo_template_parameter(parameter));
    }

    const std::map<std::string, type_data>::const_iterator type_it =
      _types.find(option.get_type());
    if (type_it == _types.end())
    {
      BOOST_THROW_EXCEPTION(illegal_state_error()
          << errinfo_template_parameter(parameter)
          << errinfo_type_name(option.get_type()));
    }
    const type_data type = type_it->second;

    stream << indent << "std::string const& var_" << stack.size() << " = "
      << "values->get_value_class_name(\"" << option.get_long_flag() << "\");"
      << std::endl;

    std::vector<class_data> const& classes = type.get_classes();
    bool first_class = true;
    for (std::vector<class_data>::const_iterator classes_it = classes.begin();
        classes_it != classes.end();
        ++classes_it)
    {
      if (!classes_it->get_template_parameters().empty())
      {
        continue; // no multiple templates
      }
      includes.insert(classes_it->get_include_path());

      stream << indent;
      if (first_class)
      {
        first_class = false;
      }
      else
      {
        stream << "else ";
      }
      stream << "if (var_" << stack.size()
        << " == \"" << classes_it->get_class_name() << "\")" << std::endl;
      stream << indent << "{" << std::endl;
      stack.push_back(classes_it->get_class_name());

      stream << create_template_class(
          class_name,
          data,
          has_type_class,
          type_class_name,
          current + 1,
          end,
          stack,
          includes);

      stack.pop_back();
      stream << indent << "}" << std::endl;
    }

    if (!first_class)
    {
      stream << indent << "else" << std::endl;
    }
    stream << indent << "{" << std::endl;
    stream << indent << "BOOST_THROW_EXCEPTION(invalid_value_error()"
      << std::endl;
    stream << indent << "    << errinfo_option_flag(\"" << option.get_long_flag()
      << "\")" << std::endl;
    stream << indent << "    << errinfo_option_value(var_" << stack.size()
      << "));" << std::endl;
    stream << indent << "}" << std::endl;
  }
  return stream.str();
}

std::string
mack::options::doc_parser::parser::create_option(
    option_data const& data,
    std::set<std::string>& includes) const
{
  std::stringstream stream;

  if (data.is_option_switch())
  {
    stream << "          new mack::options::option_switch(" << std::endl;
    stream << "              \"" << data.get_short_flag() << "\"," << std::endl;
    stream << "              \"" << data.get_long_flag() << "\"," << std::endl;
    stream << "              \""
      << replace_line_endings(data.get_description()) << "\"," << std::endl;
    stream << "              \"\"" << std::endl;
    stream << "              )" << std::endl;
  }
  else if (data.is_type_option())
  {
    const std::string type =
      std::string("mack::options::types::") + data.get_type();
    includes.insert(
        std::string("mack/options/types/") + data.get_type() + ".hpp");

    stream << "          new mack::options::type_option(" << std::endl;
    stream << "              \"" << data.get_short_flag() << "\"," << std::endl;
    stream << "              \"" << data.get_long_flag() << "\"," << std::endl;
    stream << "              \""
      << replace_line_endings(data.get_description()) << "\"," << std::endl;
    stream << "              \"\"," << std::endl;
    if (data.is_for_template_parameter())
    {
      stream << "              true," << std::endl;
    }
    else
    {
      stream << "              false," << std::endl;
    }
    if (data.has_default_value())
    {
      stream << "              new " << type << "," << std::endl;
      stream << "              \"" << data.get_default_value() << "\""
        << std::endl;
    }
    else
    {
      stream << "              new " << type << std::endl;
    }
    stream << "              )" << std::endl;
  }
  else
  {
    stream << "          new mack::options::option(" << std::endl;
    stream << "              \"" << data.get_short_flag() << "\"," << std::endl;
    stream << "              \"" << data.get_long_flag() << "\"," << std::endl;
    stream << "              \""
      << replace_line_endings(data.get_description()) << "\"," << std::endl;
    if (data.has_default_value())
    {
      stream << "              \"\"," << std::endl;
      stream << "              \"" << data.get_default_value() << "\""
        << std::endl;
    }
    else
    {
      stream << "              \"\"" << std::endl;
    }
    stream << "              )" << std::endl;
  }

  return stream.str();
}

