#ifndef __MACK_OPTIONS_DOC_PARSER_FILE_HANDLER_HPP__
#define __MACK_OPTIONS_DOC_PARSER_FILE_HANDLER_HPP__

#include <string>
#include <set>

#include <boost/filesystem.hpp>

#include <mack/options/doc_parser/parser.hpp>

namespace mack {
namespace options {
namespace doc_parser {

class file_handler
{
  public:

    file_handler(
        boost::filesystem::path const& src_directory,
        boost::filesystem::path const& xml_directory);

    ~file_handler();

    void
    run();

    void
    update_programs_file() const;

    void
    update_type_files();

  private:

    void
    delete_old_files();

    void
    check_update_file(
        std::string const& file_name,
        std::string const& content);

    void
    update_file(
        boost::filesystem::path const& file_path,
        std::string const& content) const;

    boost::filesystem::path _programs_file_path;

    boost::filesystem::path _types_directory_path;

    parser _parser;

    std::set<std::string> _old_type_files;

}; // class file_handler

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_FILE_HANDLER_HPP__ */

