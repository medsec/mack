#ifndef __MACK_OPTIONS_DOC_PARSER_OPTION_TYPES_PARSER__
#define __MACK_OPTIONS_DOC_PARSER_OPTION_TYPES_PARSER__

#include <mack/core/xml_parser.hpp>
#include <mack/options/doc_parser/data_structs.hpp>
#include <map>

namespace mack {
namespace options {
namespace doc_parser {

class option_types_parser : public mack::core::xml_parser
{
  public:

    option_types_parser(
        boost::filesystem::path const& src_directory,
        boost::filesystem::path const& xml_directory);

    virtual
    ~option_types_parser();

    virtual
    void
    clear();

    std::vector<type_data> const&
    get_child_type_datas() const;

  protected:

    virtual
    void
    start_element(
        std::string const& name,
        mack::core::attributes_type const& attributes);

    virtual
    void
    end_element(
        std::string const& name);

    virtual
    void
    characters(
        std::string const& text);

  private:

    boost::filesystem::path _src_dir;

    boost::filesystem::path _xml_dir;

    std::vector<type_data> _children;

}; // class option_types_parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_OPTION_TYPES_PARSER__ */

