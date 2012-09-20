#ifndef __MACK_OPTIONS_DOC_PARSER_DATA_STRUCTS_HPP__
#define __MACK_OPTIONS_DOC_PARSER_DATA_STRUCTS_HPP__

#include <string>
#include <vector>
#include <exception>
#include <boost/exception/exception.hpp>
#include <boost/exception/error_info.hpp>
#include <boost/exception/info.hpp>
#include <iostream>

namespace mack {
namespace options {
namespace doc_parser {

struct no_such_value_error : virtual std::exception, virtual boost::exception
{
};

struct no_option_for_tparam_error
  : virtual std::exception, virtual boost::exception
{
};

struct illegal_state_error : virtual std::exception, virtual boost::exception
{
};

typedef boost::error_info<struct tag_tparam, std::string>
  errinfo_template_parameter;

typedef boost::error_info<struct tag_type, std::string>
  errinfo_type_name;

class option_data
{
  public:

    option_data();

    ~option_data();

    std::string const&
    get_description() const;

    std::string const&
    get_short_flag() const;

    std::string const&
    get_long_flag() const;

    bool
    has_default_value() const;

    std::string const&
    get_default_value() const;

    bool
    is_option_switch() const;

    bool
    is_type_option() const;

    std::string const&
    get_type() const;

    bool
    is_for_template_parameter() const;

    std::string const&
    get_corresponding_template_parameter() const;

    friend class class_data_parser;

    friend class program_data_parser;

  private:

    void
    append_description(
        std::string const& description);

    void
    set_short_flag(
        std::string const& short_flag);

    void
    set_long_flag(
        std::string const& long_flag);

    void
    set_default_value(
        std::string const& default_value);

    void
    set_as_option_switch();

    void
    set_type(
        std::string const& type);

    void
    set_corresponding_template_parameter(
        std::string const& template_parameter);

    void
    trim_description();

    std::string _description;

    std::string _short_flag;

    std::string _long_flag;

    bool _has_default_value;

    std::string _default_value;

    bool _is_option_switch;

    bool _is_type_option;

    std::string _type;

    bool _is_for_template_parameter;

    std::string _template_parameter;

};

class class_data
{
  public:

    class_data();

    ~class_data();
 
    std::string const&
    get_class_name() const;

    std::string const&
    get_include_path() const;

    std::string const&
    get_brief_description() const;

    std::string const&
    get_detailed_description() const;

    std::vector<std::string> const&
    get_extended_classes() const;

    std::vector<option_data> const&
    get_options() const;

    option_data const&
    get_option_for_template_parameter(
        std::string const& parameter) const;

    std::vector<std::string> const&
    get_template_parameters() const;

    friend class class_data_parser;

  private:

    void
    set_class_name(
        std::string const& class_name);

    void
    set_include_path(
        std::string const& include_path);

    void
    add_to_brief_description(
        std::string const& brief_description);

    void
    add_to_detailed_description(
        std::string const& detailed_description);

    void
    add_extended_class(
        std::string const& extended_class);

    void
    add_option(
        option_data const& option);

    void
    add_template_parameter(
        std::string const& template_parameter);

    void
    trim_description();

    std::string _class_name;

    std::string _include;

    std::string _brief;

    std::string _details;

    std::vector<std::string> _extends;

    std::vector<option_data> _options;

    std::vector<std::string> _template_parameters;
};

class type_data
{
  public:

    type_data();

    ~type_data();

    std::string const&
    get_name() const;

    bool
    has_type_class() const;

    std::string const&
    get_type_class_name() const;

    std::string const&
    get_include_path() const;

    std::vector<class_data> const&
    get_classes() const;

    friend class type_data_parser;

  private:

    void
    set_name(
        std::string const& name);

    void
    set_type_class_name(
        std::string const& type_class_name);

    void
    set_include_path(
        std::string const& include_path);

    void
    add_class_data(
        class_data const& data);

    std::string _name;

    bool _has_type_class;

    std::string _type_class_name;

    std::string _include;

    std::vector<class_data> _classes;
};

class program_data
{
  public:

    program_data();

    ~program_data();

    std::string const&
    get_name() const;

    std::string const&
    get_brief_description() const;

    std::string const&
    get_detailed_description() const;

    std::vector<option_data> const&
    get_options() const;

    friend class program_data_parser;

  private:

    void
    set_name(
        std::string const& name);

    void
    add_to_brief_description(
        std::string const& brief_description);

    void
    add_to_detailed_description(
        std::string const& detailed_description);

    void
    add_option(
        option_data const& option);

    std::string _name;

    std::string _brief;

    std::string _details;

    std::vector<option_data> _options;
};

} // namespace doc_parser
} // namespace options
} // namespace mack

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::option_data const& data);

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::class_data const& data);

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::type_data const& data);

std::ostream&
operator<<(
    std::ostream& output_stream,
    mack::options::doc_parser::program_data const& data);

#endif /* __MACK_OPTIONS_DOC_PARSER_DATA_STRUCTS_HPP__ */

