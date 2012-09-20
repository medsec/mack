#ifndef __MACK_OPTIONS_DOC_PARSER_PROGRAMS_PARSER__
#define __MACK_OPTIONS_DOC_PARSER_PROGRAMS_PARSER__

#include <mack/core/xml_parser.hpp>
#include <mack/options/doc_parser/data_structs.hpp>
#include <map>

namespace mack {
namespace options {
namespace doc_parser {

class programs_parser : public mack::core::xml_parser
{
  public:

    programs_parser(
        boost::filesystem::path const& xml_directory);

    virtual
    ~programs_parser();

    virtual
    void
    clear();

    std::vector<program_data> const&
    get_programs() const;

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

    boost::filesystem::path _xml_dir;

    std::vector<program_data> _children;

}; // class programs_parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_PROGRAMS_PARSER__ */

