#ifndef __MACK_OPTIONS_DOC_PARSER_PARSER_HPP__
#define __MACK_OPTIONS_DOC_PARSER_PARSER_HPP__

#include <string>
#include <vector>
#include <set>
#include <map>

#include <boost/filesystem.hpp>

#include <mack/options/doc_parser/data_structs.hpp>

namespace mack {
namespace options {
namespace doc_parser {

class type_files_content
{

  public:

    type_files_content(
        std::string const& type_name,
        std::string const& header_file,
        std::string const& source_file,
        const bool contains_cuda_includes);

    ~type_files_content();

    std::string const&
    get_type_name() const;

    std::string const&
    get_header_content() const;

    std::string const&
    get_source_content() const;

    std::string
    get_header_file_name() const;

    std::string
    get_source_file_name() const;

  private:

    type_files_content();

    std::string _type_name;

    std::string _header_file;

    std::string _source_file;

    bool _contains_cuda_includes;
};

class parser
{
  public:

    parser(
        boost::filesystem::path const& src_directory,
        boost::filesystem::path const& xml_directory);

    ~parser();

    std::string
    create_programs_content() const;

    std::vector<type_files_content>
    create_type_files_contents() const;

  private:

    parser();

    std::string
    create_programs_content_includes(
        std::set<std::string>& includes) const;

    std::string
    create_programs_content_body(
        std::set<std::string>& includes) const;

    std::string
    create_program_content(
        program_data const& data,
        std::set<std::string>& includes) const;

    type_files_content
    create_type_files_content(
        type_data const& data) const;

    std::string
    create_type_file_header(
        type_data const& data) const;

    std::string
    create_type_file_source(
        type_data const& data,
        bool& contains_cuda_includes) const;

    std::string
    create_type_file_source_includes(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_type_file_source_body(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_type_file_source_constructor(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_type_file_source_destructor(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_type_file_source_get_class_names(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_type_file_source_get_options(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_type_file_source_create_value(
        type_data const& data,
        std::set<std::string>& includes) const;

    std::string
    create_template_class(
        std::string const& class_name,
        class_data const& data,
        const bool has_type_class,
        std::string const& type_class_name,
        std::vector<std::string>::const_iterator current,
        std::vector<std::string>::const_iterator const& end,
        std::vector<std::string>& stack,
        std::set<std::string>& includes) const;

    std::string
    create_option(
        option_data const& data,
        std::set<std::string>& includes) const;


    std::map<std::string, type_data> _types;
    std::vector<program_data> _programs;

}; // class parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_PARSER_HPP__ */

